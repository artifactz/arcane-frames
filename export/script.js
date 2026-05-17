// DOM Elements
const currentImageElement = document.getElementById('image-current');
const nextImageElement = document.getElementById('image-next');

const imagePaths = window.FILENAMES;
let currentImageIndex = Math.floor(Math.random() * imagePaths.length);

/**
 * Performs a cross-fade blend between the current and next images. Sched
 */
function crossFadeBlend(desiredBlendDuration, defaultDuration = 10000, minDuration = 300, maxDuration = 6500, assumedFps = 24.0) {
    blendDuration = show(nextImageElement, desiredBlendDuration);

    // Wait for the fade-in to complete
    setTimeout(() => {
        currentImageElement.src = nextImageElement.src;
        hide(nextImageElement);
        currentImageIndex = (currentImageIndex + 1) % imagePaths.length;
        nextImageElement.src = imagePaths[currentImageIndex];

        const currentFrame = getFrameIndex(currentImageElement.src);
        const nextFrame = getFrameIndex(nextImageElement.src);

        let displayDuration = (currentFrame < nextFrame)
            ? 1000.0 * (nextFrame - currentFrame) / assumedFps
            : defaultDuration;  // First frame of new episode
        displayDuration = Math.max(minDuration, Math.min(maxDuration, displayDuration));

        let nextBlendDuration = 0.2 * displayDuration;
        displayDuration -= nextBlendDuration;

        // Schedule next transition
        setTimeout(function() { crossFadeBlend(nextBlendDuration); }, displayDuration);
    }, blendDuration);
}

function getFrameIndex(filename) {
    return filename.split('/').pop().split('_').pop().split('.')[0];
}

/**
 * Shows the element with a fade-in effect based on the specified transition duration.
 * Returns the actual duration of the fade-in effect applied to the element.
 */
function show(element, transitionDuration) {
    if (transitionDuration >= 2000) {
        element.classList.add("visible-slow");
        return 2000;
    } else if (transitionDuration >= 1000) {
        element.classList.add("visible");
        return 1000;
    } else if (transitionDuration >= 500) {
        element.classList.add("visible-fast");
        return 500;
    } else {
        element.classList.add("visible-veryfast");
        return 250;
    }
}

/**
 * Hides the element instantly.
 */
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
    nextImageElement.src = imagePaths[currentImageIndex];
    crossFadeBlend();
}
