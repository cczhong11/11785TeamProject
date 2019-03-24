# untar
for file in videos/*.rar
do
    unrar "$file" videos
done

# convert to image
for d in $(find videos/* -maxdepth 1 -type d)
do
    for video in $d/*
    do
        mkdir "${video%.*}"
        ffmpeg -i $video "${video%.*}"/thumb%04d.jpg -hide_banner
    done
done
