# Matlab机器学习入门

## 1、机器学习是什么

## 2.Classification workflow

### 2.1、Overview

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWCtt9PBLVJzsXXS7PB%2Fimage.png?alt=media&token=e6b8a9c5-a3e6-48c4-9eb4-82ef694a8826)

### 2.2、Import Data(导入数据)

用到的matlab**函式**

```matlab
readtable("M.txt");
plot();
axis equal;	%使横纵比一样
```

M.txt类似如下:

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MWA-GZLZ2E1fSJBVFuO%2Fimage.png?alt=media&token=fb07cee1-ee93-4856-82c3-ecc024a39382)

### 2.3、Process Data(处理数据)



### 2.4、Extract Features(特征提取)

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MW9qwVW2rm0DXLBcrXh%2Fimage.png?alt=media&token=4880cb3c-2173-4bfd-b1fd-80e5efae1b76)

通过计算不同的**特征**来辨别字母

- 例如J和M，J会窄一点，而M近似于square，所以可以通过Y轴范围和X轴范围之比来确定
- 相比于J和M，V会写的快一点，所以可以把duration当作特征值来辨别

在这里我们计算两个特征

- letter的**长宽比**：AspectRatio
- letter的书写时间：Duration

```matlab
AspectRatio = range(letter.Y)/range(letter.X);
Duration = letter.Time(end) - letter.Time(1);

%range(x),  x is an array
%means  max(x)-min(x)
```

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MW9xb07Md3fVLBSv6mc%2Fimage.png?alt=media&token=8a3c4934-fec4-454a-9ee8-c084a4ccfe09)

featuredata.mat:

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MW9xuOfF9U_hunA6ppe%2Fimage.png?alt=media&token=1657ac61-063c-4fb4-943b-9c62a79e329b)

通过scatter和gscatter两个函式来visualization（散点图）

```matlab
load featuredata.mat
features

scatter(features.AspectRatio,features.Duration);

%gscatter 是可以通过第三个parameter来展现不同的颜色
gscatter(features.AspectRatio,features.Duration,features.Character)
```

scatter:

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MW9yaOH9rR8dZp8brQB%2Fimage.png?alt=media&token=3354a409-0bfd-46db-bd28-d1635f80fc58)



gscatter:

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MW9yok3MR7Pc4rX1YR2%2Fimage.png?alt=media&token=3bade2e4-3d5f-4342-b4c5-90895433565e)



### 2.5、Build a Model

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MWA-gyVr-8v5uSShMuQ%2Fimage.png?alt=media&token=e56f2131-5fad-466c-bcbd-9638fbc91678)

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MWA034-jqpJp2V1Fh5w%2Fimage.png?alt=media&token=00bfcb2a-58f3-40ab-aeb8-ae4a0d06dfe7)

在本例中，我们应用**KNN模型**，并用matlab里面自带的KNN分类器

- 关于KNN（https://zhuanlan.zhihu.com/p/25994179）

加载table

```matlab
load featuredata.mat
features	%features包括有两个维度的特征以及相应的			%label，这里的label是Character
```



![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWCcYbKNxrX4N2IKmhr%2Fimage.png?alt=media&token=c6dce066-dbc0-4a85-9364-04b4e449f96c)

用matlab的函式构造模型并预测

```matlab
knnmodel = fitcknn(features, "Character");
newdata = [4,1.2];	%parameter1是AspectRatio ...
predicted = predict(knnmodel, newdata);
```

调整K邻近的K参数

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWCctioyvvUi59HEe3S%2Fimage.png?alt=media&token=96d5afb6-cc41-45b8-b3d3-dd86955f4dc1)

预测值

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWCd4zgPkAPeoMywzVr%2Fimage.png?alt=media&token=81189e2a-b819-4475-b562-9db5c730f84a)



### 2.6、Evaluate Model（评价模型）

在本例中，我们有三种evaluate的方式

- accuracy
- misclassrate
- confusion matrix（混淆矩阵）

我们通过training data得到**model**

再通过test data来evaluate 我们的model好坏

这里的**混淆矩阵**用到了matlab的函式

```matlab
%ytrue is a vector of the known classes
%ypred is a vector of the predicted classes
confusionchart(ytrue, ypred);
```

**confusion matrix**

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWClKWP0ROe9qSFYcLb%2Fimage.png?alt=media&token=9fc4d0e8-d6a7-462e-a48a-28212e44af56)

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWClPVlCMpkuk73TlGv%2Fimage.png?alt=media&token=3b3d9d8b-ee5a-4968-88ba-c4e50443ff51)



### 2.7、Review

这次我们用的是2个特征，来辨别13个字母

我们把所有流程回顾

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWCsh8fLyWsxEjzHAXp%2Fimage.png?alt=media&token=9f99cd0a-39b8-4921-8603-bbcf4958c353)

```matlab
load featuredata13letters.mat
features
testdata

gscatter(features.AspectRatio,features.Duration,features.Character)
xlim([0 10])

knnmodel = fitcknn(features,"Character","NumNeighbors",5);
predictions = predict(knnmodel,testdata);

misclass = sum(predictions ~= testdata.Character)/numel(predictions)
confusionchart(testdata.Character,predictions);
```

gscatter:

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWCtH05fvcnZO76z-V3%2Fimage.png?alt=media&token=ab135178-b868-4f62-9952-cdc436aaf817)



confusion matrix

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MWCcUfTaZBSjR9wy8iS%2F-MWCtRXvpMNJlwu5WvzX%2Fimage.png?alt=media&token=2560f014-f3eb-4d84-9eee-c125bff3b338)