<p align="center">
  <em>ThoraxNet : A 3D U-Net based Two-stage Framework for OAR Segmentation on Thoracic CT Images</em>
</p>

## What it is?
Cancer is one of the deadliest, yet one of the most common diseases in the world. Radiation therapy is an image-guided
method for treating cancer,which uses intense radiation beams to kill cancerous cells. Therefore, generating an 
efficient and accurate treatment plan is essential in the radiation therapy workflow, which distributes a precise
radiation dose to the cancerous anatomical region and nearby tissues, referred to as organs-at-risk (OARs).
OARs are usually sensitive organs and must be safeguarded during the radiation procedure to avoid side effects. 
So OAR segmentation is a vital step in treatment planning, which can provide the size, shape, and location of the
organs in the CT images. ThoraxNet is an automatic segmentation of organs-at-risk (OARs) of thoracic organs used for radiation treatment
planning to decrease human efforts and errors.

## How it works?
We propose a two-stage deep learning-based segmentation model with an attention mechanism that automatically
delineates OARs in thoracic CT images. After preprocessing the input CT volume, a 3D U-Net architecture
will locate each organ to generate cropped images for the segmentation network. Next, two differently configured 
U-Net-based networks will perform the segmentation of large organs - left lung, right lung, heart, and small organs
esophagus and spinal cord, respectively. A post-processing step integrates all the individually-segmented organs 
to generate the final result. The trained model is developed into an end to end command line tool which helps to predicts
the organ masks of a respective CT scan and save the rt struct file. Users need to only input the original CT image series
in dicom format and the tool uses the trained MultiSegNet to output corresponding.

## Commands
`final.py` runs inference on a variety of sources, weights and saving rt struct to a destination.
```bash
$ python final.py --source [dicom series]      
                 --dest [saving path]
```               
## Contact

**Issues should be raised directly in the repository.** For professional support requests email at allenjohnbinu@gmail.com ,seenia_p190029cs@nitc.ac.in, manuthomas233@gmail.com or ajaytjose@gmail.com.
