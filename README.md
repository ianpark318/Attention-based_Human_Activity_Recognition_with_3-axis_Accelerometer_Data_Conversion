# Attention-based Human Activity Recognition with 3-axis Accelerometer Data Conversion
`3축 가속도 데이터 변환 및 Attention 기반 사람 행동인식`  

A pytorch code about ETRI2023,  
**Attention-based Human Activity Recognition with 3-axis Accelerometer Data Conversion**.  
  
  
To **train** or **inference** our models, please clone this repository.😀

This project was researched by [Minseong Kweon](https://github.com/mnseong), [Jaehyeong Park](https://github.com/ianpark318), [Kyunghyun Kim](https://github.com/Ga-ng), [Jeonghyun Noh](https://github.com/JJeong-Gari)   

Feel free to contact us if you have any questions,  
📬 wou1202@pusan.ac.kr  
📬 ianpark318@pusan.ac.kr  
📬 klps44@pusan.ac.kr  
📬 wjdgus0967@pusan.ac.kr  
<br><br>
![image](/img/image1.png)
<br><br>
___
# Data Preprocessing
- `Convert_RP.py` converts time series datas of 3-axis accelerometer to RP. <br>
- `MFCC_convert.ipynb` converts time series datas of 3-axis accelerometer to MFCC. <br>
___
# Training Model

```
$ ./run_mfcc.sh
$ ./run_rp.sh
```
- `train_mfcc.py` trains mfcc images <br>
- `train_rp` trains rp images <br>
___
# Download pretrained models
Download `pth` files [here](https://drive.google.com/drive/u/0/folders/1ng7q5NMGdmWgQY4FZBjv9YmftLXnbAjh)
___
# Inference
```
$ ./infer.sh
```
- `infer.py` tests our model (inference) <br>
___
# Paper Reference
[1] Ranasinghe, S., AI Machot, F., Mayr, H. C, “A review on applications of activity recognition systems with regard to performance and evaluation”, Internal Journal of Distributed Sensor Network, vol. 12 no. 8, 2016. <br><br>
[2] Jaeyoung Chang, et al, “Development of Real-time Video Surveillance System Using the Intelligent Behavior Recognition Technique”, The Journal of The Institute of Internet, Broadcasting and Communication, vol. 19, no. 2, pp. 161-168, 2020. <br><br>
[3] Nedorubova, A., Kadyrova, A., Khlyupin, A., “Human activity recognition using continuous wavelet transform and convolutional neural network”, doi: https://doi.org/10.48550/arXiv.2106.12666, 2021. <br><br>
[4] Chen, Y., Xue, Y., “A deep learning approach to human activity recognition based on single accelerometer”, In 2015 IEEE international conference onsystems, man, and cybernetics, pp. 1488-1492, 2015. <br><br>
[5] He, Z., He, Z., “Accelerometer-based Gesture Recognition Using MFCC and HMM”, In 2018 IEEE 4th International Conference on Computer and Communications (ICCC), pp. 1435-1439, 2018. <br><br>
[6] Seungeun Chung, et al., “Real-world multimodallifelog dataset for human behavior study”, ETRI Journal, vol. 43, no. 6, 2021.
[7] Jianjie, L., Kai-Yu, Tong, “Robust Single Accelerometer-Based Activity Recognition Using Modified Recurrence Plot”, IEEE Sensors Journal, vol. 19, no. 15, 2019. <br><br>

