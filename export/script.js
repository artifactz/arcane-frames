// DOM Elements
const currentImageElement = document.getElementById('image-current');
const nextImageElement = document.getElementById('image-next');

let currentImageIndex = 0;
const imagePaths = window.FILENAMES;

function crossFadeBlend() {
    const currentFrame = getFrameIndex(currentImageElement.src);
    const nextFrame = getFrameIndex(nextImageElement.src);
    let displayDuration = 1000.0 * (nextFrame - currentFrame) / 24.0;
    let blendDuration = 0.2 * displayDuration;

    blendDuration = show(nextImageElement, blendDuration);
    displayDuration -= blendDuration;

    // Wait for the fade out time
    setTimeout(() => {
        currentImageElement.src = nextImageElement.src;
        hide(nextImageElement);
        nextImageElement.src = imagePaths[++currentImageIndex];

        // Schedule next transition
        setTimeout(crossFadeBlend, displayDuration);
    }, blendDuration);
}

function getFrameIndex(filename) {
    return filename.split('/').pop().split('_').pop().split('.')[0];
}

function show(element, transitionDuration) {
    if (transitionDuration > 2000) {
        element.classList.add("visible-slow");
        return 2000;
    } else if (transitionDuration > 1000) {
        element.classList.add("visible");
        return 1000;
    } else if (transitionDuration > 500) {
        element.classList.add("visible-fast");
        return 500;
    } else {
        element.classList.add("visible-veryfast");
        return 250;
    }
}

function hide(element) {
    element.classList.remove("visible");
    element.classList.remove("visible-slow");
    element.classList.remove("visible-fast");
    element.classList.remove("visible-veryfast");
}

// Initialization
if (imagePaths.length < 2) {
    alert("Please add at least two images to the imagePaths array to demonstrate blending.");
    currentImageElement.style.opacity = '0';
    nextImageElement.style.opacity = '0';
} else {
    // Start the first cycle
    nextImageElement.src = imagePaths[0];
    crossFadeBlend();
}
