#/bin/sh

# the file is hosted on https://github.com/NEWSLabNTU/wayside-release/
sha1sum -c 2021-03-28-12-02-57.728+0800_4582000_04.74864.ckpt.sha1 || \
    (
        aria2c \
            -x4 \
            -c \
            -V \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --checksum=sha-1=2b4dd12ed0f9925c7cb91f078b11777c6944baf4 \
            'https://github.com/NEWSLabNTU/wayside-release/releases/download/0.1.0/2021-03-28-12-02-57.728+0800_4582000_04.74864.ckpt.xz' && \
            unxz --force --verbose 2021-03-28-12-02-57.728+0800_4582000_04.74864.ckpt.xz
    )

sha1sum -c yolov4-csp-custom-128x128-2021-07-14_2021-07-15-08-36-58.242+0800_186000_02.22891.ckpt.sha1 || \
    (
        aria2c \
            -x4 \
            -c \
            -V \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --checksum=sha-1=5779f5bab63246f4a261e2d1c60400ca3d6a97fe \
            'https://github.com/NEWSLabNTU/wayside-release/releases/download/0.1.1/yolov4-csp-custom-128x128-2021-07-14_2021-07-15-08-36-58.242+0800_186000_02.22891.ckpt'
    )
