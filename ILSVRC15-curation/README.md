### Step-by-step instructions to create your own curated ILSVRC15 dataset

Project page: <http://www.robots.ox.ac.uk/~luca/siamese-fc.html>

![crops image][logo]

[logo]: http://www.robots.ox.ac.uk/~luca/stuff/siamese-fc_pairs.jpg "Crops image"

1. Download and unzip the full (the 86 GB archive) original ImageNet Video dataset.
2. Move `ILSVRC15/Data/VID/validation` to `ILSVRC15/Data/VID/train/` so that inside `train/` there are 5 folders with the same structure. It is a good idea to rename these folders and use very short names ( I have used a, b, c, d, e) in order to save some bytes in the metadata.
3. Run `./video_ids.sh /path/to/original/ILSVRC2015` to generate `vid_id_frames.txt`.
4. Run `per_frame_annotation.m` for all 5 folders.
5. Run `parse_objects.m` for all 5 folders.
6. Run `vid_setup_data.m` to generate your own `imdb_video.mat`.
7. Duplicate the tree structure of the ILSVRC15 folder (without copying the data!), using something like [this](http://stackoverflow.com/questions/4073969/copy-folder-structure-sans-files-from-one-location-to-another) (for UNIX) or [this](http://superuser.com/questions/530128/how-to-copy-a-directory-structure-without-copying-files) (for Windows). This folder will contain your curated dataset.
8. Run `save_crops.m` to generate crops for all the videos indexed by `imdb_video.mat` and save them on disk.
9. Run `vid_image_stats.m` and save the output in a `.mat` file. It will be used for data augmentation during training.
10. All set! You can now continue from point 2.v. from `./siamese-fc/README.md` and train the network with your custom curated ILSVRC15!
