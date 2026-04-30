# arcane-frames

This project extracts frames from videos (such as the Netflix show *Arcane*) and generates a slideshow as a Desktop wallpaper in HTML format.

## "Pipeline" right now

* You need video files in the `episodes/` subfolder.
* Run `index.py` to build the database.
* Run `extract_images.py` to populate `export/images/`.
* Run `export.py` to update `export/filenames.js`.
* Use `Lively Wallpaper`, `WallpaperWebPage`, `Wallpaper Engine`, or similar to set `export/index.html` as a wallpaper.
