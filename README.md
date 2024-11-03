# LowRCAS-ViT: 
This project aims to add adapters to the proposed architecture in 

https://github.com/Tianfang-Zhang/CAS-ViT

Thank you for the public repository and your incredible contributions

Before any execution add the necessary files following https://github.com/Tianfang-Zhang/CAS-ViT, as well as the network weights or some dataset

## Run in collab
```bash
!git clone https://github.com/JorgeCecatto/LowRCAS-ViT.git
```
```bash
main_dir = "/content/LowRCAS-ViT/classification"
os.chdir(main_dir)
```
```bash
python fine_tune.py --data_path path_to_split_in_training_test_validation --batch_size 32 --input_size 384 --finetune path_to_weights --lr 5e-5 --nb_classes 2 --model rcvit_xs --adapters True
```

## Run in container 🐙

```bash
docker build --tag image_name .
```
```bash
docker run --it image_name docker_name 
```
It is a base image, based on running on a cpu, to use a gpu consider changing the image from python to pytorch and ensure you have a version of the nvidia tool kit installed

After that, continue execution and add the necessary files following https://github.com/Tianfang-Zhang/CAS-ViT, as well as the network weights or some dataset

to run by adding adapters

```bash
python fine_tune.py --data_path path_to_split_in_training_test_validation --batch_size 32 --input_size 384 --finetune path_to_weights --lr 5e-5 --nb_classes 2 --model rcvit_xs --adapters True
```

If you have problems with the cv2 package, run the following commands
```bash
pip uninstall opencv-python
```
and 
```bash
pip install opencv-python-headless
```

# Project in progress 🚀