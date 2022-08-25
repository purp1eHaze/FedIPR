# FedIPR-Repository

[TPAMI 2022](https://ieeexplore.ieee.org/document/9847383) | [ArXiv](https://arxiv.org/abs/2109.13236) | [PDF](https://arxiv.org/pdf/2109.13236.pdf)

### Official pytorch implementation of the paper: 
#### - FedIPR: Ownership Verification for Federated Deep Neural Network Models

Accepted by [TPAMI 2022](https://ieeexplore.ieee.org/document/9847383)

## Description

<p align="justify"> Federated learning models are collaboratively developed upon valuable training data owned by multiple parties. During the
development and deployment of federated models, they are exposed to risks including illegal copying, re-distribution, misuse and/or
free-riding. To address these risks, the ownership verification of federated learning models is a prerequisite that protects federated
learning model intellectual property rights (IPR) i.e., FedIPR. We propose a novel federated deep neural network (FedDNN) ownership
verification scheme that allows private watermarks to be embedded and verified to claim legitimate IPR of FedDNN models. In the
proposed scheme, each client independently verifies the existence of the model watermarks and claims respective ownership of the
federated model without disclosing neither private training data nor private watermark information. </p>

![avatar](imgs/Framwork_new.eps)

<p align="center"> Figure 1: Framework of FedIPR </p>

## How to run

## Dataset

## How to embed feature-based watermarks into a desired layer

All configs are stored in `/`

For example, a layer with 256 channels, so the maximum will be 256-bit === 32 ascii characters are allowed. If the watermark is less than 32 characters, the remaining bits will be set randomly.

The example below is AlexNet with the last 3 layers as the passport layer, i.e we embed random signature into the 4th and 5th layer and embed `this is mine` into the last layer (6th).

```
{
  "0": false,
  "2": false,
  "4": true,
  "5": true,
  "6": "this is mine"
}
```

## Citation
If you find this work useful for your research, please cite
```
TBD
```
&#169;2022 Webank and Shanghai Jiao Tong University.
