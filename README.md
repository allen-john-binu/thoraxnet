<p align="center">
  <em>ThoraxNet : Multi Model approach to Thoracic CT Image segmentation</em>
</p>

## What it is?
Automatic segmentation of organs-at-risk (OARs) of thoracic organs used for radiation treatment planning to decrease human efforts and errors. It is done using yoloV5 and 3D Unet.

## How it works?
The trained model is developed into an end to end command line tool which helps to predicts the organ masks of a respective CT scan and save the rt struct file.  Users need to only input the original CT image series in dicom format and the tool uses the trained MultiSegNet to output corresponding.

## Commands
`mask.py` runs inference on a variety of sources, weights and saving rt struct to a destination.
```bash
$ python final.py --source [dicom series]      
                 --dest [saving path]
```               
## Contact

**Issues should be raised directly in the repository.** For professional support requests email at allenjohnbinu@gmail.com , manuthomas233@gmail.com or ajaytjose@gmail.com.
