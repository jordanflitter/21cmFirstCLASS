# 21cmFirstCLASS
This is an extension of the popular `21cmFAST` code that interfaces with `CLASS` to generate initial conditions at recombination that are consistent with the input cosmological model. These initial conditions can be set during the time of recombination, allowing one to compute the 21cm signal (and its spatial fluctuations) throughout the dark ages, as well as in the proceeding cosmic dawn and reionization epochs, just like in the standard `21cmFAST`.

## Small taste of what can be done with the code
`21cmFirstCLASS` tracks both the CDM density field $\delta_c$ as well as the baryons density field $\delta_b$.

![densities](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/images/densities.png)

`21cmFirstCLASS` allows you to consistently compute the brightness temperature field at the dark ages, as well as in the cosmic dawn and reionization epochs (like in `21cmFAST`).

![coeval boxes](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/images/coeval_boxes.png)
![lightcone boxes](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/images/lightcone_boxes.png)

In addition, the user interface in `21cmFirstCLASS` has been improved and allows one to easily plot the 21cm power spectrum while including noise from the output of `21cmSense`.
![power spectrum](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/images/power_spectrum.png)

## Using the code
Comprehensive jupyter notebook tutorials have been made for this code, check them out at the following links.
* [Notebook #1](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/Tutorial%20(Jupyter%20notebooks)/notebook_1.ipynb) for installation instructions and basic usage.
* [Notebook #2](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/Tutorial%20(Jupyter%20notebooks)/notebook_2.ipynb) for learning about the new physical features (in &Lambda;CDM cosmology) that have been introduced to `21cmFirstCLASS`, like running the simulation through the dark ages, evolving the baryon density field, and more.
* [Notebook #3](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/Tutorial%20(Jupyter%20notebooks)/notebook_3.ipynb) for studying beyond &Lambda;CDM models with `21cmFirstCLASS`.
* [Notebook #4](https://github.com/jordanflitter/21cmFirstCLASS/blob/main/Tutorial%20(Jupyter%20notebooks)/notebook_4.ipynb) for studying the detectability of the 21cm signal with `21cmSense`.

## Acknowledging
`21cmFirstCLASS` is an open source code and you are encouraged to use it for your studies. If you use this code please cite:
* Jordan Flitter and Ely D. Kovetz, _"New tool for 21-cm cosmology. I. Probing &Lambda;CDM and beyond"_, Phys. Rev. D 109 (2024) 4, 043512 ([arXiv: 2309.03942](https://arxiv.org/pdf/2309.03942)).
* Jordan Flitter and Ely D. Kovetz, _"New tool for 21-cm cosmology. II. Investigating the effect of early linear fluctuations"_, Phys. Rev. D 109 (2024) 4, 043513 ([arXiv: 2309.03948](https://arxiv.org/pdf/2309.03948)).

As `21cmFirstCLASS` is based on [21cmFAST](https://github.com/21cmfast/21cmFAST/tree/master), please also cite the associated 21cmFAST papers.
* Andrei Mesinger, Steven Furlanetto and Renyue Cen, _"21CMFAST: a fast, seminumerical simulation of the high-redshift 21-cm signal"_, Mon. Not. Roy. Astron. Soc. 411 (2011) 955 ([arXiv: 1003.3878](https://arxiv.org/pdf/1003.3878)).
* Muñoz, J.B., Qin, Y., Mesinger, A., Murray, S., Greig, B., and Mason, C., _"The Impact of the First Galaxies on Cosmic Dawn and Reionization"_, Mon. Not. Roy. Astron. Soc. 511 (2022) 3, 3657-3681 ([arXiv: 2110.13919](https://arxiv.org/pdf/2110.13919)).

Moreover, besides of incorporating new features in `21cmFAST`, `21cmFirstCLASS` also integrates various open source codes. Make sure you cite the relevant papers from the following github links if you use `21cmFirstCLASS` to...
* Generate consistent initial conditions for the `21cmFAST` simulation (or to perform a joint 21cm-CMB analysis) with [CLASS](https://github.com/lesgourg/class_public).
* Compute precisely the free electron fraction with [HYREC](https://github.com/nanoomlee/HYREC-2).
* Compute the power spectrum of any box with [powerbox](https://github.com/steven-murray/powerbox).
* Compute the noise of 21cm interferometers like HERA with [21cmSense](https://github.com/rasg-affiliates/21cmSense).
* Study fuzzy dark matter with [AxiCLASS](https://github.com/PoulinV/AxiCLASS).
* Study scattering dark matter with [dmeff-CLASS](https://github.com/kboddy/class_public/tree/dmeff).
