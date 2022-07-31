# FedIPR-Repository

[ArXiv](https://arxiv.org/abs/2109.13236) | [PDF](https://arxiv.org/pdf/2109.13236.pdf)

### Official pytorch implementation of the paper: 
#### - FedIPR: Ownership Verification for Federated Deep Neural Network Models

Accepted by [TPAMI 2022]()

## Description

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
