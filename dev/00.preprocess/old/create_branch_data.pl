# POD documentation - main docs before the code

=head1 NAME

Name - Short description

=head1 SYNOPSIS

This script gives the xyz of the ... using zzz. It can be used as ... to do sxxs.

=head1 Create Branch Data

This script takes a file of genomic loci (e.g. bed) and a reference (e.g. fasta) and produces an array per locus with the information of the reference 

=cut

# Let the code begin...


#!/usr/bin/env perl

use Modern::Perl;
use autodie;
use Getopt::Long::Descriptive;

# Define and read command line options
my ($opt, $usage) = describe_options(
	"Usage: %c %o",
	["create_branch_data.pl - Creates branch data from list of loci and reference"],
	[],
	['ifile=s', 'Input file name. Ignore for STDIN', {default => "-"}],
	['itype=i', 'Input file type. Default: bed. Supported: [bed]', {default => "bed"}],
	['rfile=s', 'Reference file name. Ignore for STDIN', {default => "-"}],
	['rtype=i', 'Reference file type. Default: fasta. Supported: [fasta, fa]', {default => "fasta"}],
	['ofile=s', 'Output file name. Ignore for STDOUT', {default => "-"}],
	['verbose|v', 'Print progress'],
	['help|h', 'Print usage and exit',
		{shortcircuit => 1}],
);
print($usage->text), exit if $opt->help;

# Parameter checks
if (($opt->ifile eq "-") and ($opt->rfile eq "-"){
    die "Cannot read both input and reference from STDIN. Please define one of them as a file source\n";
}

# Reading regions
if ($opt->verbose){warn "Reading regions\n";}
my $regions;
if (lc($opt->itype) eq "bed"){
    $regions = GenOO::RegionCollection::Factory->create('BED', {
        file => $opt->ifile
    })->read_collection;
}

# Opening Reference
if ($opt->verbose){warn "Reading reference file\n";}
my $ref;
if ((lc($opt->rtype) eq "fasta") or (lc($opt->rtype) eq "fa")){
    $fp = GenOO::Data::File::FASTA->new(file => $opt->rfile);
}




#Opening filehandles


my $OUT = filehandle_for($opt->ofile, "out");







exit;

##############################
##############################

sub filehandle_for {
#   this opens a filehande by filename or opens STDIN if the filename is "-"
	my ($file, $inout) = @_;

	if ((!defined $inout) or (lc($inout) eq "in")){
        if (($file eq '-'){
            open(my $IN, "<-");
            return $IN;
        }
        else {
            open(my $IN, "<", $file);
            return $IN;
        }
    }
    else {
        if (($file eq '-'){
            open(my $OUT, ">");
            return $OUT;
        }
        else {
            open(my $OUT, ">", $file);
            return $OUT;
        }
    }
}


sub region_sequence_from_seq {
	my ($seq_ref, $strand, $rname, $start, $stop, $flank) = @_;

	#out of bounds
	return if ($start - $flank < 0);
	return if ($stop + $flank > length($$seq_ref) - 1);

	if ($strand == 1) {
		return substr($$seq_ref, $start-$flank, 2*$flank+$stop-$start+1);
	}
	else {
		my $seq = reverse(
			substr($$seq_ref, $start-$flank, 2*$flank+$stop-$start+1));
		if ($seq =~ /U/i) {
			$seq =~ tr/ATGCUatgcu/UACGAuacga/;
		}
		else {
			$seq =~ tr/ATGCUatgcu/TACGAtacga/;
		}
		return $seq;
	}
}

sub trim_to_size {
	my ($seq, $length) = @_;
	
	return undef if (length($seq) < $length);
	
	while (length($seq) >= $length+2){
		#trim on both sides
		my @seq = split('', $seq);
		pop @seq;
		shift @seq;
		$seq = join('', @seq);		
	}
	
	if (length($seq) > $length){
		# if there is 1 nt left to trim we will do it randomly either from the beginning or end
		my @seq = split('', $seq);
		if (rand() > 0.5){pop @seq;}
		else {shift @seq;}
		$seq = join('', @seq);	
	}
	
	return $seq;
}
