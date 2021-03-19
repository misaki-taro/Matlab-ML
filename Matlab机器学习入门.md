# Matlab机器学习入门

## 1、机器学习是什么

## 2.2、Import Data(导入数据)

用到的matlab**函式**

```matlab
readtable("M.txt");
plot();
axis equal;	%使横纵比一样
```

M.txt类似如下:

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MWA-GZLZ2E1fSJBVFuO%2Fimage.png?alt=media&token=fb07cee1-ee93-4856-82c3-ecc024a39382)

## 2.3、Process Data(处理数据)



## 2.4、Extract Features(特征提取)

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



## 2.5、Build a Model

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MWA-gyVr-8v5uSShMuQ%2Fimage.png?alt=media&token=e56f2131-5fad-466c-bcbd-9638fbc91678)

![img](https://gblobscdn.gitbook.com/assets%2F-MW9qnt5mI5Jcz-Mufx8%2F-MW9qqp3ZbOQzVBLR-40%2F-MWA034-jqpJp2V1Fh5w%2Fimage.png?alt=media&token=00bfcb2a-58f3-40ab-aeb8-ae4a0d06dfe7)

