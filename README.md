# semantic segmentation project

## 프로젝트 개요
![](https://s3-us-west-2.amazonaws.com/aistages-prod-server-public/app/Users/00000274/files/260df78e-93d2-49c4-92e7-5a070237063e..png)

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

- **Input :** hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 json file로 제공
- **Output :** 모델은 각 픽셀 좌표에 따른 class를 출력하고, 이를 rle로 변환하여 리턴합니다. 이를 output 양식에 맞게 csv 파일을 만들어 제출


|  이름      | 역할                                                         | github                         |
| :-------: | ------------------------------------------------------------ | ------------------------------ |
|김용환       |                                         | https://github.com/96namsan |
|김우진       |              | https://github.com/    |
|신건희       |                  | https://github.com/Rigel0718   |
|신중현       |                        | https://github.com/Blackeyes0u0    |
|이종휘       |                | https://github.com/gndldl    |



# Contents

```
baseline
├── eda
│   ├── stratified k fold validation.ipynb
│   └── eda.ipynb
│   
├── mmdetection
│   ├── config
│   ├── comfusion matrix
│   ├── output image.ipynb
│   ├── wbf emsemble.ipynb
│   ├── ensemble_confidence.ipynb
│   ├── train.py
│   └── inference.py
│
├── requirements.txt
│
└── yolo
		├── dataset
		│   ├── images
		│   └── labels
		└── cocotrash.yaml
```