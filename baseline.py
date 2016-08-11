import os, sys
from itertools import izip
import gzip
import tempfile
import hashlib
from subprocess import Popen, PIPE
from collections import namedtuple

import numpy as np
from scipy.stats import itemfreq
from scipy.stats.mstats import mquantiles

import h5py

import pybedtools

import pysam
import pandas as pd

from bw import BigWig

from pyDNAbinding.binding_model import DNASequence, PWMBindingModel, DNABindingModels, load_binding_models
from pyDNAbinding.DB import (
    load_binding_models_from_db, NoBindingModelsFoundError, load_all_pwms_from_db)

GenomicRegion = namedtuple('GenomicRegion', ['contig', 'start', 'stop'])

DNASE_IDR_PEAKS_BASE_DIR = "..."
DNASE_FOLD_COV_DIR = "..."
TRAIN_TSV_BASE_DIR = "..."
TEST_TSV_BASE_DIR = "..."

aggregate_region_scores_labels = [
    "mean", "max", "q99", "q95", "q90", "q75", "q50"]
def aggregate_region_scores(
        scores, quantile_probs = [0.99, 0.95, 0.90, 0.75, 0.50]):
    rv = [scores.mean()/len(scores), scores.max()]
    rv.extend(mquantiles(scores, prob=quantile_probs))
    return rv

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_model(factor_name):
    """Load models from the DB.

    This isn't useful - I just keep it here to show where the models came from.
    """
    try:
        models = load_binding_models_from_db(tf_names=[factor_name,])
        assert len(models) == 1, "Multiple binding models found for '{}'".format(factor_name)
    except NoBindingModelsFoundError:
        # if we couldnt find a good motif, just find any motif
        # special case TAF1 because it doesnt exist in CISBP
        if factor_name == 'TAF1':
            models = [load_TAF1_binding_model(),]
        else:
            models = load_all_pwms_from_db(tf_names=factor_name)
    model = models[0]
    return model

def save_models():
    """Save models from all factors

    This isn't useful - I just keep it here to show where the models came from.
    """
    factor_names = """
    ARID3A
    ATF2
    ATF3
    ATF7
    CEBPB
    CREB1
    CTCF
    E2F1
    E2F6
    EGR1
    EP300
    FOXA1
    FOXA2
    GABPA
    GATA3
    HNF4A
    JUND
    MAFK
    MAX
    MYC
    NANOG
    REST
    RFX5
    SPI1
    SRF
    STAT3
    TAF1
    TCF12
    TCF7L2
    TEAD4
    YY1
    ZNF143
    """.split()
    mos = []
    for factor_name in factor_names:
        mos.append(load_model(factor_name))
    mos = DNABindingModels(mos)
    with open("models.yaml", "w") as ofp:
        ofp.write(mos.yaml_str)


class LabelData(object):
    def __len__(self):
        return len(self.data)
    
    @property
    def samples(self):
        return self.data.columns.values

    def build_integer_labels(self, label_index):
        labels = np.zeros(self.data.ix[:,label_index].shape, dtype=int)
        labels[np.array(self.data.ix[:,label_index] == 'A', dtype=bool)] = -1
        labels[np.array(self.data.ix[:,label_index] == 'B', dtype=bool)] = 1
        return labels

    def build_train_array(self, normalize=True):
        """Unstack the data frame accessibility score and labels.

        """
        predictors = []
        for i in xrange(len(self.samples)):
            predictors.append(self.dataframe.ix[:,i])
        return
    
    def iter_regions(self, flank_size=0):
        for contig, start, stop in self.data.index:
            yield GenomicRegion(contig, start, stop)
        return
    
    def iter_seqs(self, fasta_fname):
        genome = pysam.FastaFile(fasta_fname)
        return (
            genome.fetch(contig, start, stop+1).upper()
            for contig, start, stop
            in self.iter_regions(flank_size=400)
        )

    def _init_header_data(self, labels_fname):
        with gzip.open(labels_fname) as fp:
            header_line = next(iter(fp))
        header_data = header_line.strip().split("\t")
        if header_data[:3] != ['chr', 'start', 'stop']:
            raise ValueError(
                "Unrecognized header line: '%s'" % header_line.strip())
        self.header = header_data
        return

    def __hash__(self):
        if self._hash is None:
            self._hash = abs(hash(hashlib.md5(str((
                md5(self.labels_fname),
                None if self.regions_fname is None else md5(self.regions_fname),
                self.max_n_rows
            ))).hexdigest()))
        return self._hash

    @property
    def cached_fname(self):
        return "labeldata.%s.%s.obj" % (self.factor, hash(self))

    def _build_dataframe(self):
        # if filter_regions is specified, then restrict the labels to
        # regions that overlap these
        if self.regions_fname is not None:
            # load the data into a bed file
            # zcat {fname} | tail -n +2 | head -n 10000 | \
            #   bedtools intersect -wa -a stdin -b {regions_fname} \
            # | uniq > {output_fname}
            filtered_regions_fp = tempfile.NamedTemporaryFile("w+")
            p1 = Popen(["zcat", self.labels_fname], stdout=PIPE)
            p2 = Popen(["tail", "-n", "+2",], stdout=PIPE, stdin=p1.stdout)
            # check to see if we should limit the numbere of input rows
            p4_input = None
            # if we want to limit the number of rows, then add a call to head
            if self.max_n_rows is not None:
                p3 = Popen(
                    ["head", "-n", str(self.max_n_rows),], stdout=PIPE, stdin=p2.stdout)
                p4_input = p3.stdout
            else:
                p3 = None
                p4_input = p2.stdout
            p4 = Popen(["bedtools", "intersect",
                        "-wa",
                        "-a", "stdin",
                        "-b", self.regions_fname],
                       stdin=p4_input,
                       stdout=PIPE
            )
            p5 = Popen(["uniq",], stdin=p4.stdout, stdout=filtered_regions_fp)
            p5.wait()
            # Allow p* to receive a SIGPIPE if p(*-1) exits.
            p1.terminate()  
            p2.terminate()
            if p3 is not None: p3.terminate()
            p4.terminate()
            # flush the output file cache, and reset the file pointer
            filtered_regions_fp.flush()
            filtered_regions_fp.seek(0)
            return pd.read_table(
                filtered_regions_fp,
                index_col=(0,1,2),
                nrows=self.max_n_rows,
                names=self.header)
        else:
            return pd.read_table(
                self.labels_fname,
                header=0,
                index_col=(0,1,2),
                nrows=self.max_n_rows
            )
    
    def __init__(self,
                 labels_fname,
                 regions_fname=None,
                 max_n_rows=None,
                 load_cached=True):
        self.labels_fname = labels_fname
        self.regions_fname = regions_fname
        self.max_n_rows = max_n_rows
        self._hash = None
        self.load_cached = load_cached
        # extract the sample names from the header
        #assert labels_fname.endswith("labels.tsv.gz"), \
        #    "Unrecognized labels filename '%s'" % labels_fname
        self._init_header_data(labels_fname)
        # extract the factor from the filename
        self.factor = os.path.basename(labels_fname).split('.')[0]

        # if we want to use a cached version...
        if self.load_cached is True:
            try:
                print "Loading '%s'" % self.cached_fname
                self.h5store = h5py.File(self.cached_fname)
                self.data = pd.read_hdf(self.cached_fname, 'data')
            except KeyError:
                self.data = self._build_dataframe()
                self.data.to_hdf(self.cached_fname, 'data')
                print self.h5store
        else:
            self.data = self._build_dataframe()
        
        return
    
    def build_motif_scores(self, fasta_fname):
        all_agg_scores = np.zeros(
            (len(self), len(aggregate_region_scores_labels)), dtype=float)
        binding_models = load_binding_models("models.yaml")
        model = binding_models.get_from_tfname(self.factor)
        for i, seq in enumerate(self.iter_seqs(fasta_fname)):
            if i%1000000 == 0: print >> sys.stderr, i, len(self)
            all_agg_scores[i,:] = aggregate_region_scores(
                DNASequence(seq).score_binding_sites(model, 'MAX')
            )        
        all_agg_scores = pd.DataFrame(
            all_agg_scores,
            index=self.data.index,
            columns=aggregate_region_scores_labels)
        return all_agg_scores

    def load_or_build_motif_scores(self, fasta_fname):
        try:
            self.motif_scores = pd.read_hdf(self.cached_fname, 'motif_scores')
            self.motif_scores.index = self.data.index
        except KeyError:
            self.motif_scores = self.build_motif_scores(fasta_fname)
            self.motif_scores.to_hdf(self.cached_fname, 'motif_scores')
        return self.motif_scores

    def build_dnase_fc_scores(self):
        path=DNASE_FOLD_COV_DIR
        scores = np.zeros((len(self), len(self.samples)), dtype=float)
        for sample_i, sample_name in enumerate(self.samples):
            fname = "DNASE.{}.fc.signal.bigwig".format(sample_name)
            b = BigWig(os.path.join(path, fname))
            for region_i, region in enumerate(self.iter_regions()):
                if region_i%1000000 == 0:
                    print "Sample %i/%i, row %i/%i" % (
                        sample_i+1, len(self.samples), region_i, len(self))
                scores[region_i, sample_i] = b.stats(
                    region.contig, region.start, region.stop, 'mean')
            b.close()
        return pd.DataFrame(
            np.nan_to_num(scores), columns=self.samples, index=self.data.index)
    
    def load_or_build_dnase_fc_scores(self):
        try:
            self.dnase_fc_scores = pd.read_hdf(self.cached_fname, 'dnase_scores')
        except KeyError:
            self.dnase_fc_scores = self.build_dnase_fc_scores()
            self.dnase_fc_scores.to_hdf(self.cached_fname, 'dnase_scores')
        except IOError:
            self.dnase_fc_scores = self.build_dnase_fc_scores()            
        return self.dnase_fc_scores

    def dnase_dataframe(self, normalize=True):
        dnase_scores = self.load_or_build_dnase_fc_scores()
        if normalize:
            dnase_scores = (dnase_scores - dnase_scores.mean())/dnase_scores.std()
        dnase_scores = pd.concat([
            dnase_scores[column] for column in dnase_scores.columns
        ], axis=0, ignore_index=True)
        dnase_scores.columns = ['dnase_score',]
        return dnase_scores
        
    def dataframe(self, fasta_fname, normalize=True):
        # load and normalzie the motif and DNASE scores
        motif_scores = self.load_or_build_motif_scores(fasta_fname)
        if normalize:
            motif_scores = (motif_scores - motif_scores.mean())/motif_scores.std()

        # concat the DNASE scores into a single column, and repeat the motif
        # scores so that each DNASE entry has a single motif score entry
        motif_scores = pd.concat(
            [motif_scores]*len(dnase_scores.columns), axis=0, ignore_index=True)
        # concat the dnase scores and motif scores, and then rename the columns
        rv = pd.concat([motif_scores, dnase_scores], axis=1)
        rv.columns = list(motif_scores.columns) + ['dnase_score',]
        return rv

def train_model(factor):
    """Train a simple model to predict in-vivo binding.
    
    """
    labels_fname = TRAIN_TSV_BASE_DIR + "{}.train.labels.tsv.gz".format(factor)
    all_data = LabelData(labels_fname)
    sample_labels = []
    subset_train_data = []
    subset_train_labels = []
    for sample_i, sample in enumerate(all_data.samples):
        print sample_i, sample
        sample_labels.append(sample)
        dnase_peaks_fname = (
            DNASE_IDR_PEAKS_BASE_DIR 
            + "DNASE.%s.conservative.narrowPeak.gz" % sample
        )
        sample_data = LabelData(labels_fname, dnase_peaks_fname)
        dnase_scores = sample_data.load_or_build_dnase_fc_scores()[[sample,]]
        dnase_scores.columns = ['dnase_fc',]
        motif_scores = all_data.load_or_build_motif_scores('hg19.genome.fa')
        sample_train_df = dnase_scores.join(motif_scores)
        subset_train_data.append(sample_train_df)
        sample_train_labels = sample_data.build_integer_labels(sample_i)
        subset_train_labels.append(sample_train_labels)

    print "Building the training data sets"
    train_df = pd.concat(subset_train_data, levels=sample_labels)
    print train_df.head()
    train_labels = np.concatenate(subset_train_labels, axis=0)

    print "Filtering out ambiguous labels"
    non_ambiguous_labels = (train_labels > -0.5)
    train_amb_filtered_mat = train_df.iloc[non_ambiguous_labels].as_matrix()
    train_amb_filtered_labels = train_labels[non_ambiguous_labels]

    print "Fitting the model"
    from sklearn.linear_model import SGDClassifier
    mo = SGDClassifier(
        loss='log', class_weight='balanced', n_jobs=16)
    mo.fit(train_amb_filtered_mat, train_amb_filtered_labels)

    print "Loading the test set"
    true_labels_fname = FULL_GENOME_TSV_BASE_DIR + \
        "{}.train.labels.tsv.gz".format(factor)
    label_data = LabelData(true_labels_fname)
    print label_data.samples
    for sample in label_data.samples:
        print "Loading the test predictors"
        dnase_peaks_fname = (
            DNASE_IDR_PEAKS_BASE_DIR 
            + "DNASE.%s.conservative.narrowPeak.gz" % sample
        )
        sample_data = LabelData(true_labels_fname, dnase_peaks_fname)
        dnase_scores = sample_data.load_or_build_dnase_fc_scores()[[sample,]]
        dnase_scores.columns = ['dnase_fc',]
        print "Building the test data dataframe"
        motif_scores = label_data.load_or_build_motif_scores('hg19.genome.fa')
        sample_train_df = dnase_scores.join(motif_scores, how='inner')
        print sample_train_df.head()
        print "Predicting prbs"
        pred_prbs = mo.predict_proba(sample_train_df)[:,1]
        result = pd.DataFrame({'prb': pred_prbs}, index=sample_train_df.index)
        result = result.reindex(label_data.data.index, fill_value=0.0)
        print result.head()

        ofname = 'F.{}.{}.tab'.format(factor, sample)
        result.to_csv(ofname, sep="\t", header=False)
    return

def main():
    train_model(sys.argv[1])

if __name__ == '__main__':
    main()
