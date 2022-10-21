# Reduced Order Model (ROM) using CNN
This directory holds reduced order models using convolutional neural network.

Framework : Tensorflow 2.5.8 with CUDA 11.2 and CUDNN

![image](https://user-images.githubusercontent.com/16720947/179875607-ed424ec3-868c-4ce7-acae-17c05b3f3d24.png)

- CNN에 관한 요약: (참고-http://taewan.kim/post/cnn/)
1) Input: 입력변수. 전체 변수영역을 Input feature map라고 부름
2) Kernel: Input feature map을 돌아다니는 그림자 사각요소. Input과의 연산을 통해 Output feature map을 만듦
3) Channel: 입력변수의 갯수
4) Stride: 커널이 움직이는 간격
5) Zero padding: Feature map 둘레로 0값을 가지는 Layer 갯수
6) Pooling: 커널을 사용한 map 추출 시 평균값 혹은 최대값만 뽑아서 사이즈를 줄이는 기법
7) Convolution, Deconvolution: 변수로부터 Feature를 추출하는 과정은 Convolution, 그 반대 과정은 Deconvolution (혹은 transposed convolution) 

![image](https://user-images.githubusercontent.com/16720947/179884429-7aceeaa5-23ce-4f17-be55-c9b38185cf9c.png)


- Convolution
 
![Convolution_arithmetic_-_No_padding_no_strides](https://user-images.githubusercontent.com/16720947/179883147-a1cd71f7-13f0-4266-8e59-e8bc44c7edcc.gif)

- Deconvolution


![YyCu2](https://user-images.githubusercontent.com/16720947/179883158-9fb7a660-fda8-42d9-8eec-bf1b90b87dc6.gif)

N.B.: Blue maps are inputs, and cyan maps are outputs.

[애니메이션 참고 - https://github.com/vdumoulin/conv_arithmetic]
