# BINDER

Here, we present BINDER for accurately and robustly identifying hierarchical TADs from Hi-C data. Based on the hypothesis that the anchoring of TAD boundaries is a key feature of TADs, BINDER comprehensively generates consensus TAD boundaries and yields hierarchical TADs.

The workflow of BINDER is as follows.

## Requirements for installation

**1.** Python 3.10.2

**2.** torch 2.2.2

**3.** numpy 1.23.5

**4.** scipy 1.11.0

## Installation

**1.** Unzip "BINDER-master.zip" you have downloaded:

`$ unzip BINDER-master.zip -d BINDER`

**2.** Then enter the BINDER folder：

`$ cd BINDER`

**3.** Add execution permission for necessary program 'Infomap':

`$ chmod +x Infomap`

## Usage of BINDER
		
    python BINDER.py [options] -m <hic_file> -r <resolution> -chr <chromosome>

**Required**

    --matrix/-m <string>          : Path to N x N raw Hi-C matrix;

    --resolution/-r <int>         : resolution of Hi-C matrix (kb);

    --chromosome/-chr <string>    : chromosome of Hi-C matrix;

**Optional**

    --output/-o <string>          : Output folder, the output result is saved in the BINDER_result folder by default;

**Typical commands**

The following command is an example:

    python BINDER.py -m example_data/GM12878_50kb_chr22.txt -r 50 -chr chr22

**Output**

(i) The result is saved in BINDER_result/Result.txt by default.

(ii) An example output is shown below (resolution=50kb, chr=chr22):

    Left_position	Right_position	Level	Type
    17600000	18050000	2	Domain
    17600000	18250000	1	Domain
    17600000	18350000	0	Domain

The first and second columns indicate the left and right positions (bases) of TAD, the third column indicates the hierarchy of TAD (level of gaps is "non-level"), and the fourth column indicates whether the base pair interval is domain or gap.

**Changelog**

The current version of BINDER is 1.0, subsequnt version will be updated.
 
