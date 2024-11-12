# CM-Bio
CM-Bio：a unified programming paradigm that spans computational and microbial systems

## Background
We propose a unified programming paradigm that spans computational and microbial systems, leveraging their intrinsic digital properties. As a proof of concept, we have developed the “CM-Bio” program suite, which achieves data encryption at a level over 100 orders of magnitude higher than previous strategies. This integration of programming across the computational-biological divide represents a significant advancement in information encryption, potentially opening up new avenues for future social security.

## Requirement
* numpy: 1.26.0
* opencv-python: 4.5.5.64

## Usage
* **encode**  
```
python encryption.py -i minecraft.png -o test -c config.json
```
> -i: file for encrypt, a colorful image
> -o: output file  
> -c: config file which include two numbers of encode tables

* **decode**  
```
python decryption.py -i .\test\out_chest.png -o out -c config.json
```
> -i: encrypted image
> -o: output directory  
> -c: config file which include two numbers of encode tables
