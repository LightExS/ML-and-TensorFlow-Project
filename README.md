This programm should run by without any editing as long as all libraries are installed. Pictures that are neccessary for this wil be downloaded automatically in "C:\Users\USER_NAME\.keras\datasets" folder

Most of us libraries used in this project are preinstalled with python. Those that not preinstalled will be listed below.
-Tensorflow version: 2.12.0

For testing purposes you can either create and train model by yourself (which takes 12 minutes on my pc) or load already trained model.
To choose one of them comment/uncomment last lines of code



For testing i used pictures that model never saw and 9 of 10 times the guess was right. Detailed results are listed below.

## Daisy 
https://www.thespruce.com/thmb/-wN_FsvmZMoMC3yi5hg_5EIJXcU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/oxeye-daisy-growing-guide-5190951-hero-baed472653934a6da8c8f86237dcf7bc.jpg ---- recognizes as daisy with 100%

https://cdn.britannica.com/36/82536-050-7E968918/Shasta-daisies.jpg ---- recognizes as daisy with 99.81%

## Dandelion
https://cdn.shopify.com/s/files/1/0016/1968/9545/articles/dmitry-tulupov-htm3AoUb5dU-unsplash_1024x1024.jpg?v=1646694818 ---- recognizes as dandelion with 99.98%

https://www.gardeningknowhow.com/wp-content/uploads/2017/08/dandelion-seed-head.jpg ---- recognizes as dandelion with 99.44%

## Roses
https://cdn11.bigcommerce.com/s-i7i23daso6/images/stencil/1280x1280/products/5214/12501/Rose_Queen_Elizabeth_0001678__67691.1623341668.jpg?c=1 ---- recognizes as roses with 98.09%

https://hips.hearstapps.com/hmg-prod/images/rose-color-meanings-1674960738.jpg ---- recognizes as roses with 81.20%

## Tulips
https://cdn11.bigcommerce.com/s-1b9100svju/product_images/uploaded_images/mixed-tulips.jpg ---- recognizes as tulips with 99.73%

https://storage.googleapis.com/pod_public/1300/127233.jpg ---- recognizes as tulips with 98.99%

## Sunflowers
https://images.squarespace-cdn.com/content/v1/56bf55504c2f85a60a9b9fe5/1635897793784-OD3181KEQJ2AV5QTEEK6/SunflowerSunset.jpg?format=1000w ---- recognizes as dandelion with 61.29%

https://images.unsplash.com/photo-1597848212624-a19eb35e2651?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8MXx8fGVufDB8fHx8&w=1000&q=80 ---- recognizes as sunflower with 93.05%
