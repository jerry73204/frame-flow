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
