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

    --normalization/-n <string>   : normalization method: 'SCN' [1], 'ICE' [2], 'KR' [3], 'sqrtVC' [1], default='SCN';

**Optional**

    --output/-o <string>          : Output folder, the output result is saved in the BINDER_result folder by default;

**Typical commands**

The following command is an example:

    python BINDER.py -m example_data/GM12878_50kb_chr22.txt -r 50 -chr chr22 -n SCN

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

## References
**1.** SCN/sqrtVC: Zhang, S., Krieger, J. M., Zhang, Y., Kaya, C., Kaynak, B., Mikulska-Ruminska, K., Doruker, P., Li, H., & Bahar, I. (2021). ProDy 2.0: increased scale and scope after 10 years of protein dynamics modelling with Python. Bioinformatics (Oxford, England), 37(20), 3657–3659. https://doi.org/10.1093/bioinformatics/btab187
**2.** ICE: Servant, N., Varoquaux, N., Lajoie, B. R., Viara, E., Chen, C. J., Vert, J. P., Heard, E., Dekker, J., & Barillot, E. (2015). HiC-Pro: an optimized and flexible pipeline for Hi-C data processing. Genome biology, 16, 259. https://doi.org/10.1186/s13059-015-0831-x
**3.** KR：Kumar, R., Sobhy, H., Stenberg, P., & Lizana, L. (2017). Genome contact map explorer: a platform for the comparison, interactive visualization and analysis of genome contact maps. Nucleic acids research, 45(17), e152. https://doi.org/10.1093/nar/gkx644

