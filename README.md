![colortransfer_example](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/928791b0-b734-4835-92c9-cdcb12fcddc7)
# ColorTransferLib
![python3.10.12](https://img.shields.io/badge/build-3.10.12-blue?logo=python&label=Python) ![](https://img.shields.io/badge/build-22.04.3%20LTS-orange?logo=ubuntu&label=Ubuntu
) ![](https://img.shields.io/badge/build-MIT-purple?label=License) ![](https://img.shields.io/badge/build-6.4.0-brown?logo=octave&label=Octave) ![](https://img.shields.io/badge/build-GeForce%20RTX%203060-white?logo=nvidia&label=GPU) ![](https://img.shields.io/badge/build-intel%20Core%20i7--13700KF-white?logo=intel&label=CPU)

The ColorTransferLib is a library focused on color transfer, featuring a range of published algorithms. Some algorithms have been re-implemented, while others are integrated from public repositories. The primary objective of this project is to compile all existing color and style transfer methods into one library with a standardized API. This aids the research community in both development and comparison of algorithms. Currently, the library supports 15 color and style transfer methods for images, 3D point clouds, and textured triangle meshes. Additionally, it includes 20 metrics for evaluating color transfer results. A detailed list of all algorithms is available below.

![compatability](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/d59972fd-e135-4572-8682-d3f5f8f85c75)

## API
For seamless integration, adhere to the API specifications of the new color transfer algorithm, depicted in the Figure below.

Each class demands three inputs: Source, Reference, and Options. The Source and Reference should be of the **Image** or **Mesh** class type, with the latter encompassing 3D point clouds and textured triangle meshes. The Options input consists of dictionaries, stored as a JSON file in the **Options** folder. For a sample option, see Listings 1. Every option details an adjustable parameter for the algorithm.

Save each new color transfer class in the ColorTransferLib Repository under the **Algorithms** folder. This ensures its automatic integration into the user interface. The class should have two essential functions: **get_info()** and **apply(...)**. The **get_info()** function yields vital details about the algorithm (refer to Listing 2). It also provides data type details, facilitating the identification of compatible objects for the algorithm. The **apply(...)** function ingests the inputs and embodies the core logic for color transfer.

The output should resemble a dictionary format, as outlined in Listing 3. A status code of 0 signifies a valid algorithm output, while -1 indicates invalidity. The process time denotes the algorithm's execution duration, useful for subsequent evaluations. The 'object' key in the dictionary holds the result, which should match the class type of the Source input.

![CT-API](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/7e59eea8-78be-4dfb-acae-7e8abfd7abe5)

![Listing](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/93692741-46f6-4955-80a5-d152fa22104d)

## Installation
### Requirements
Download the Models-Folder from the following link and place it in the project's root directory.
```
...
```

Run the following commands once:
```
sudo apt-get install liboctave-dev
pkg install -forge image
pkg install -forge statistics
```
- To enable VSI
```
user@pc:~/<Project Path>/ColorTransferLib/Evaluation/VIS/gbvs$ ocatve
octave:1> gbvs_install.m
```


### Install via PyPI
TODO

### Install from source
```
pip install -r requirements/requirements.txt
python setup.py bdist_wheel
pip install --force-reinstall ../ColorTransferLib/dist/ColorTransferLib-0.0.2-py3-none-any.whl 
```

## Available Color Transfer Methods:
The following color and style transfer methods are integrated in the library. Some of them are reimplemented based on the algorithm's description in the the published papers and others are adopted from existing repositories and adpated to fit the API. The original implementation of the latter methods can be found next to the publication's name. The superscripts (2D, 3D) indicated wether the algorithm is applicable to 2D structures like images and mesh textures or to 3D structures like point clouds. The subscript (CT, ST) describs wether the algorithm is a color or a style transfer algorithm.

| Year | ID  | Publication |
| ---  | --- | --- |
| 2001 | $`GLO^{2D,3D}_{CT}`$ | [Color Transfer between Images](https://doi.org/10.1109/38.946629) |
| 2003 | $`BCC^{2D,3D}_{CT}`$ | [A Framework for Transfer Colors Based on the Basic Color Categories](https://doi.org/10.1109/CGI.2003.1214463) |
| 2005 | $`PDF^{2D,3D}_{CT}`$ | [N-dimensional probability density function transfer and its application to color transfer](https://doi.org/10.1109/ICCV.2005.166) |
| 2006 | $`CCS^{2D,3D}_{CT}`$ | [Color transfer in correlated color space](https://doi.org/10.1145/1128923.1128974) |
| 2007 | $`MKL^{2D,3D}_{CT}`$ | [The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer](https://doi.org/10.1049/cp:20070055) |
| 2009 | $`GPC^{2D}_{CT}`$ | [Gradient-Preserving Color Transfer](http://dx.doi.org/10.1111/j.1467-8659.2009.01566.x) |
| 2010 | $`FUZ^{2D,3D}_{CT}`$ | [An efficient fuzzy clustering-based color transfer method](https://doi.org/10.1109/FSKD.2010.5569560) |
| 2015 | $`NST^{2D}_{ST}`$ | [A Neural Algorithm of Artistic Style](https://doi.org/10.48550/arXiv.1508.06576) - [Original Implementation](https://github.com/cysmith/neural-style-tf) |
| 2017 | $`DPT^{2D}_{ST}`$ | [Deep Photo Style Transfer](https://doi.org/10.48550/arXiv.1703.07511) - [Original Implementation](https://github.com/LouieYang/deep-photo-styletransfer-tf) |
| 2019 | $`TPS^{2D}_{CT}`$ | [L2 Divergence for robust colour transfer](https://doi.org/10.1016/j.cviu.2019.02.002) - [Original Implementation](https://github.com/groganma/gmm-colour-transfer) |
| 2020 | $`HIS^{2D}_{CT}`$ | [Deep Color Transfer using Histogram Analogy](https://doi.org/10.1007/s00371-020-01921-6) - [Original Implementation](https://github.com/codeslake/Color_Transfer_Histogram_Analogy) |
| 2020 | $`PSN^{3D}_{ST}`$ | [PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color](https://doi.org/10.1109/WACV45572.2020.9093513) - [Original Implementation](https://github.com/hoshino042/psnet) |
| 2020 | $`EB3^{3D}_{CT}`$ | [Example-Based Colour Transfer for 3D Point Clouds](https://doi.org/10.1111/cgf.14388) |
| 2021 | $`CAM^{2D}_{ST}`$ | [CAMS: Color-Aware Multi-Style Transfer](https://doi.org/10.48550/arXiv.2106.13920) - [Original Implementation](https://github.com/mahmoudnafifi/color-aware-style-transfer) |
| 2021 | $`RHG^{2D}_{CT}`$ | [HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms](https://doi.org/10.48550/arXiv.2011.11731) |


## Available Objective Evaluation Metrics
Three classes of evaluation metrics are considered here. Metrics that evaluate the color consistency with the reference image (indicated with $`^r`$), metrics that evaluate the structural similarity with the source image (indicated with $`^s`$) and metrics that evaluates the overall quality of the output (indicated with $`^o`$).

| Year | ID  | Name | Publication |
| ---  | --- | --- | --- |
| / | $`PSNR^s_{rgb}`$ | Peak Signal-to-Noise Ratio | / |
| / | $`HI^r_{rgb}`$ | Histogram Intersection | / |
| / | $`Corr^r_{rgb}`$ | Correlation | / |
| / | $`BA^r_{rgb}`$ | Bhattacharyya Distance | / |
| / | $`MSE^s_{rgb}`$ | Mean-Squared Error | / |
| / | $`RMSE^s_{rgb}`$ | Root-Mean-Squared Error | / |
| 2003 | $`CF^o_{rgyb}`$ | Colorfulness | [Measuring Colourfulness in Natural Images](http://dx.doi.org/10.1117/12.477378) |
| 2003 | $`MSSSIM^s_{rgb}`$ | Multi-Scale Structural Similarity Index | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) |
| 2004 | $`SSIM^s_{rgb}`$ | Structural Similarity Index | [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861) |
| 2006 | $`GSSIM^s_{rgb}`$ | Gradient-based Structural Similarity Index | [Gradient-Based Structural Similarity for Image Quality Assessment](https://doi.org/10.1109/ICIP.2006.313132) |
| 2010 | $`IVSSIM^s_{rgb}`$ | 4-component Structural Similarity Index | [Content-partitioned structural similarity index for image quality assessment](https://doi.org/10.1016/j.image.2010.03.004) |
| 2011 | $`IVEGSSIM^s_{rgb}`$ | 4-component enhanced Gradient-based Structural Similarity Index | [An image similarity measure using enhanced human visual system characteristics](https://ui.adsabs.harvard.edu/link_gateway/2011SPIE.8063E..10N/doi:10.1117/12.883301) |
| 2011 | $`FSIM^s_{c,yiq}`$ | Feature Similarity Index | [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730) |
| 2012 | $`BRISQUE^o_{rgb}`$ | Blind/Referenceless Image Spatial Quality Evaluator | [No-Reference Image Quality Assessment in the Spatial Domain](https://doi.org/10.1109/TIP.2012.2214050) |
| 2013 | $`NIQE^o_{rgb}`$ | Naturalness Image Quality Evaluator | [Making a “Completely Blind” Image Quality Analyzer](https://doi.org/10.1109/LSP.2012.2227726) |
| 2014 | $`VSI^s_{rgb}`$ | Visual Saliency-based Index | [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028) |
| 2016 | $`CTQM^{sro}_{lab}`$ | Color Transfer Quality Metric | [Novel multi-color transfer algorithms and quality measure](https://doi.org/10.1109/TCE.2016.7613196) |
| 2018 | $`LPIPS^s_{rgb}`$ | Learned Perceptual Image Patch Similarity | [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://doi.org/10.1109/CVPR.2018.00068) |
| 2018 | $`NIMA^o_{rgb}`$ | Neural Image Assessment | [NIMA: Neural Image Assessment](https://doi.org/10.48550/arXiv.1709.05424) |
| 2019 | $`CSS^{sr}_{rgb}`$ | Color and Structure Similarity | [Selective color transfer with multi-source images](https://doi.org/10.1016/j.patrec.2009.01.004) |

## Citation
If you utilize this code in your research, kindly provide a citation:
```
@inproceeding{potechius2023,
  title={A software test bed for sharing and evaluating color transfer algorithms for images and 3D objects},
  author={Herbert Potechius, Thomas Sikora, Gunasekaran Raja, Sebastian Knorr},
  year={2023},
  booktitle={European Conference on Visual Media Production (CVMP)},
  doi={10.1145/3626495.3626509}
}
```
