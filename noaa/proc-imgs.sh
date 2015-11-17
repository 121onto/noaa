#!/usr/bin/env bash

src='/your/base/directory/noaa/data/imgs/'
dst='/your/base/directory/noaa/data/imgs-proc/'

for path in $(find "$src" -name "*.jpg"); do
    file=$(basename "$path")
    convert "$path" -resize 300x300^ -gravity Center -crop 300x300+0+0 +repage "$dst$file";
    for i in `seq 0 90 270`; do
        convert "$dst$file" -virtual-pixel black -distort SRT $i "$dst""t_""$i""_""$file"
    done
    echo "$file"
done
