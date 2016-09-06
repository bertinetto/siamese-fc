#!/bin/bash
# Generate files with video ids and nframes per video.
# argument $1 is the absolute path of the root path of original ILSVRC15.
# e.g. ./video_ids.sh /path/to/original/ILSVRC2015

rm vid_id_frames.txt

rootdir=$1
datadir='ILSVRC2015/Data/VID/train'
cnt=1

for l1 in `ls $rootdir/$datadir`
do
	for l2 in `ls $rootdir/$datadir/$l1`
	do
		nframes=`ls $rootdir/$datadir/$l1/$l2 | wc -l`
		echo $l1/$l2 $cnt $nframes
		cnt=$((cnt+1))
	done
done >> vid_id_frames.txt

