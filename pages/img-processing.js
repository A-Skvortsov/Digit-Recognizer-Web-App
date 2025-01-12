import { fabric } from 'fabric';
import { canvas, CONTAINER_HEIGHT, CONTAINER_WIDTH, IMG_HEIGHT, IMG_WIDTH, RGBA} from "./index";

export function processImg() {
	const widthRatio = CONTAINER_WIDTH / IMG_WIDTH;
	const heightRatio = CONTAINER_HEIGHT / IMG_HEIGHT;
	
	const ctx = canvas.getContext('2d');
	const pixels = ctx.getImageData(0, 0, CONTAINER_WIDTH, CONTAINER_HEIGHT).data;
	const grayScale = toGrayScale(pixels)
	const image = resize(grayScale, widthRatio, heightRatio);
	
	return image;
}


function toGrayScale(arr) {
	let newArr = [];
	for (let i = 0; i < arr.length; i+=RGBA) {
		newArr.push(arr[i]);
	}
	
	return newArr;
}


function resize(originalArray, widthRatio, heightRatio) {
	const newArray = Array(IMG_WIDTH * IMG_WIDTH).fill(0);  // Initialize new 28x28 array

	  // Loop through every pixel in the new 28x28 image
	  for (let i = 0; i < IMG_WIDTH; i++) {  // i is the vertical index of the new image
	    for (let j = 0; j < IMG_WIDTH; j++) {  // j is the horizontal index of the new image
	      let pixelSum = 0;
	      let count = 0;

	      // Calculate the corresponding region in the original image
	      for (let y = i * heightRatio; y < (i + 1) * heightRatio; y++) {  // Loop through the height of the region
	        for (let x = j * widthRatio; x < (j + 1) * widthRatio; x++) {  // Loop through the width of the region
	          const originalIndex = y * CONTAINER_WIDTH + x;
	          pixelSum += originalArray[originalIndex];  // Accumulate pixel values
	          count++;
	        }
	      }

	      // Average the pixel values in the region and assign to the new array
	      const avgPixelValue = Math.round(pixelSum / count);
	      newArray[i * IMG_WIDTH + j] = avgPixelValue;
	    }
	  }

	  return newArray;
}


export default function placeHolder() {
	return <div/>;
}