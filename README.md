# Cont-Bouchand Model

This repository contains the implementation of the Cont-Bouchand model for stock price fluctuations, which I developed as part of a course project on statistical physics.

## Overview

The Cont-Bouchand model aims to explain certain qualitative characteristics of stock price fluctuations by incorporating herd behavior. Specifically, it predicts that short-term price fluctuations follow an exponentially truncated power-law distribution. You can explore the main experiments related to this model in the [cont_bouchand.ipynb](cont_bouchand.ipynb) notebook. I also gave a presentation on my results and it can be found in [Cont_Bouchand_presentation.pdf](Cont_Bouchand_presentation.pdf).

## Dynamic Extension

The original Cont-Bouchand model uses a fixed probability \( p \), which determines the likelihood that two traders will act identically. As an extension, I developed a "dynamic" version of the model, where \( p \) is treated as a random variable.

To achieve this, I modeled \( p \) using the Cox-Ingersoll-Ross (CIR) process, augmented with basic affine jump diffusion. This choice was motivated by two factors:
1. **Mean Reversion**: The CIR model includes mean-reverting behavior, reflecting the idea that \( p \) should have a typical or average value.
2. **Market Crashes and Booms**: The affine jump diffusion extension allows for occasional spikes in \( p \), representing extreme events like market crashes or booms, where a large majority of traders either sell or buy.

The experiments related to this dynamic extension can be found in the following notebooks:
- [cox_ingersoll_ross_test.ipynb](cox_ingersoll_ross_test.ipynb)
- [dynamic_cont_bouchand.ipynb](dynamic_cont_bouchand.ipynb)