#@title Licensed under the Apache License, Version 2.0
#https://www.apache.org/licenses/LICENSE-2.0

import tensorflow as tf

def generate_path_inputs(baseline, input, alphas):
    """
    Generate interpolated 'images' along a linear path at alpha intervals between a baseline tensor

    baseline: 2D, shape: (200, 4)
    input: preprocessed sample, shape: (200, 4)
    alphas: list of steps in interpolated image ,shape: (21)


    return: shape (21, 200, 4)
    """
    # Expand dimensions for vectorized computation of interpolations.
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(input, axis=0)
    delta = input_x - baseline_x
    path_inputs = baseline_x + alphas_x * delta

    return path_inputs


def compute_gradients(model, path_inputs):
    """
    compute dependency of each field on whole result, compared to interpolated 'images'

    :param model: trained model
    :param path_inputs: interpolated tensors, shape: (21, 200, 4)
    :return: shape: (21, 200, 4)
    """
    with tf.GradientTape() as tape:
        tape.watch(path_inputs)
        predictions = model(path_inputs)

        outputs = []
        for envelope in predictions:
            outputs.append(envelope[0])
        outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)

    gradients = tape.gradient(outputs, path_inputs)
    return gradients


def generate_alphas(m_steps=50, method='riemann_trapezoidal'):
    """
    Args:
    m_steps(Tensor): A 0D tensor of an int corresponding to the number of linear
    interpolation steps for computing an approximate integral. Default is 50.
    method(str): A string representing the integral approximation method. The
       following methods are implemented:
      - riemann_trapezoidal(default)
      - riemann_left
      - riemann_midpoint
      - riemann_right
    Returns:
      alphas(Tensor): A 1D tensor of uniformly spaced floats with the shape
      (m_steps,).
      """
    m_steps_float = tf.cast(m_steps, float)

    if method == 'riemann_trapezoidal':
        alphas = tf.linspace(0.0, 1.0, m_steps+1)
    elif method == 'riemann_left':
        alphas = tf.linspace(0.0, 1.0 - (1.0 / m_steps_float), m_steps)
    elif method == 'riemann_midpoint':
        alphas = tf.linspace(1.0 / (2.0 * m_steps_float), 1.0 - 1.0 / (2.0 * m_steps_float), m_steps)
    elif method == 'riemann_right':
        alphas = tf.linspace(1.0 / m_steps_float, 1.0, m_steps)
    else:
        raise AssertionError("Provided Riemann approximation method is not valid.")

    return alphas


def integral_approximation(gradients, method='riemann_trapezoidal'):
    """Compute numerical approximation of integral from gradients.
    Args:
    gradients(Tensor): A 4D tensor of floats with the shape
    (m_steps, img_height, img_width, 3).
    method(str): A string representing the integral approximation method. The
    following methods are implemented:
    - riemann_trapezoidal(default)
    - riemann_left
    - riemann_midpoint
    - riemann_right
    Returns:
    integrated_gradients(Tensor): A 3D tensor of floats with the shape
    (img_height, img_width, 3).
    """
    if method == 'riemann_trapezoidal':
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    elif method == 'riemann_left':
        grads = gradients
    elif method == 'riemann_midpoint':
        grads = gradients
    elif method == 'riemann_right':
        grads = gradients
    else:
        raise AssertionError("Provided Riemann approximation method is not valid.")

    # Average integration approximation.
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients


def integrated_gradients(model, baseline, input, m_steps=50, method='riemann_trapezoidal',
                         batch_size=32):
    """
    Args:
      model(keras.Model): A trained model to generate predictions and inspect.
      baseline(Tensor): 2D, shape: (200, 4)
      input(Tensor): preprocessed sample, shape: (200, 4)
      m_steps(Tensor): A 0D tensor of an integer corresponding to the number of
        linear interpolation steps for computing an approximate integral.
      method(str): A string representing the integral approximation method. The
        following methods are implemented:
        - riemann_trapezoidal(default)
        - riemann_left
        - riemann_midpoint
        - riemann_right
      batch_size(Tensor): A 0D tensor of an integer corresponding to a batch
        size for alpha to scale computation and prevent OOM errors. Note: needs to
        be tf.int64 and shoud be < m_steps. Default value is 32.
    Returns:
      integrated_gradients(Tensor): A 2D tensor of floats with the same
        shape as the input tensor.
    """

    # 1. Generate alphas.
    alphas = generate_alphas(m_steps=m_steps,
                             method=method)

    # Initialize TensorArray outside loop to collect gradients. Note: this data structure
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = generate_path_inputs(baseline=baseline,
                                                             input=input,
                                                             alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(model=model,
                                           path_inputs=interpolated_path_input_batch)

        # Write batch indices and gradients to TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients,
                                           method=method)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (input - baseline) * avg_gradients

    return integrated_gradients

def choose_validation_points(integrated_gradients, window_size, width):
    """
    Args:
          integrated_gradients(Tensor): A 2D tensor of floats with shape (window_size, width_of_sequence_encoded).
          window_size: int, length of sequence, num of bases
          width: int, width of encoded base
    Return: List of attributes for highlighting DNA string sequence
    """
    attr = []
    for i in range(window_size):
        for j in range(width):
            if integrated_gradients[i][j].numpy() == 0:
                continue
            attr.append(integrated_gradients[i][j].numpy())
    return attr


def visualize_token_attrs(sequence, attrs):
    """
    Visualize attributions for given set of tokens.
    Args:
    - tokens: An array of tokens
    - attrs: An array of attributions, of same size as 'tokens',
      with attrs[i] being the attribution to tokens[i]

    Returns:
    - visualization: HTML text with colorful representation of DNA sequence
        build on model prediction
    """

    def get_color(attr):
        if attr > 0:
            red = int(128 * attr) + 127
            green = 128 - int(64 * attr)
            blue = 128 - int(64 * attr)
        else:
            red = 128 + int(64 * attr)
            green = 128 + int(64 * attr)
            blue = int(-128 * attr) + 127

        return red, green, blue

    # normalize attributions for visualization.
    bound = max(abs(max(attrs)), abs(min(attrs)))
    attrs = attrs / bound
    html_text = ""
    for i, tok in enumerate(sequence):
        r, g, b = get_color(attrs[i])
        html_text += " <span style='color:rgb(%d,%d,%d)'>%s</span>" % (r, g, b, tok)

    return html_text
