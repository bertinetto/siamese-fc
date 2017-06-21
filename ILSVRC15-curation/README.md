### Step-by-step instructions to create your own curated ILSVRC15 dataset

Project page: <http://www.robots.ox.ac.uk/~luca/siamese-fc.html>

![crops image][logo]

[logo]: http://www.robots.ox.ac.uk/~luca/stuff/siamese-fc_pairs.jpg "Crops image"

1. Signup [here](http://image-net.org/challenges/LSVRC/2015/signup) to obtain the link to download the data of the 2015 challenge.
2. Download and unzip the full original ImageNet Video dataset (the 86 GB archive).
3. Move `ILSVRC15/Data/VID/validation` to `ILSVRC15/Data/VID/train/` so that inside `train/` there are 5 folders with the same structure. It is a good idea to rename these folders and use very short names ( I have used `a`, `b`, `c`, `d` and `e`) in order to save some bytes in the metadata.
4. Run `./video_ids.sh /path/to/original/ILSVRC2015` to generate `vid_id_frames.txt`.
5. Run `per_frame_annotation.m` for all 5 folders.
6. Run `parse_objects.m` for all 5 folders.
7. Run `vid_setup_data.m` to generate your own `imdb_video.mat`. Otherwise download the one we have already created - [here](http://bit.ly/imdb_video) for the one used for the ECCV'16 SiamFC, [here](http://bit.ly/cfnet_imdb_video) for the one used for the CVPR'17 CFNet.
7bis. (only for CVPR'17 CFNet code) Add a field `.set` which is 1 for ILSVRC15-VID training videos (folders `a`, `b`, `c` and `d`)and 2 for ILSVRC15-VID validation videos (folder `e`) (to not be confused with the validation videos used during tracking evaluation, which instead come from VOT and TempleColor). For ECCV'16 SiamFC the training/validation split was decided only inside `experiment.m` code.
8. Duplicate the tree structure of the ILSVRC15 folder (without copying the data!), using something like [this](http://stackoverflow.com/questions/4073969/copy-folder-structure-sans-files-from-one-location-to-another) (for UNIX) or [this](http://superuser.com/questions/530128/how-to-copy-a-directory-structure-without-copying-files) (for Windows). This folder will contain your curated dataset.
9. Run `save_crops.m` to generate crops for all the videos indexed by `imdb_video.mat` and save them on disk.
10. Run `vid_image_stats.m` and save the output in a `.mat` file. It will be used for data augmentation during training. Otherwise download the one we have already created [here](http://bit.ly/imdb_video_stats).
