# Reduced Order Model (ROM) using CNN
This directory holds reduced order models using convolutional neural network.

Framework : Tensorflow 2

![image](https://user-images.githubusercontent.com/16720947/179875607-ed424ec3-868c-4ce7-acae-17c05b3f3d24.png)

- CNNs main building blocks:
1) Input: 전체 변수영역을 Input feature map라고 부름.
2) Kernel: Input feature map을 돌아다니는 그림자 사각요소. Input과의 연산을 통해 Output feature map을 만듦.
3) Stride: 커널이 움직이는 간격
4) Zero padding: Feature map 둘레로 0값을 가지는 Layer 갯수
5) Pooling operations reduce the size of feature maps by using some function to summarize subregions, such as taking the average or the maximum value.

![image](https://user-images.githubusercontent.com/16720947/179882532-e189f4e2-c4dd-461a-9879-2f3050e76031.png)

The size of the output will be equal to the number of steps made, plus one, accounting for the initial position of the kernel, considering that the kernel starts on the leftmost part of the input feature map and slides by steps of one until it touches the right side of the input with stride 1 and without padding.

- Convolution
- 
![Convolution_arithmetic_-_No_padding_no_strides](https://user-images.githubusercontent.com/16720947/179883147-a1cd71f7-13f0-4266-8e59-e8bc44c7edcc.gif)

- Deconvolution


![YyCu2](https://user-images.githubusercontent.com/16720947/179883158-9fb7a660-fda8-42d9-8eec-bf1b90b87dc6.gif)

[참고 - https://github.com/vdumoulin/conv_arithmetic]
