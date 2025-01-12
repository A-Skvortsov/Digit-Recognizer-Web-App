import Head from "next/head";
import { processImg } from "./img-processing"
import styles from "@/styles/Home.module.css";
import { useEffect, useRef } from 'react';
import { fabric } from 'fabric';

export var canvas;
const CANV_HEIGHT = 196;  //size of canvas in pixels
const CANV_WIDTH = 196;
export const CONTAINER_HEIGHT = 2 * CANV_HEIGHT;  //actual screen size of the canvas container
export const CONTAINER_WIDTH = 2 * CANV_WIDTH;
export const IMG_WIDTH = 28;
export const IMG_HEIGHT = 28;
export const RGBA = 4;  //# of values representing a pixel in RGBA format

let resultRef;
const BRUSH_SIZE = 15;


export default function Home() {
	resultRef = useRef(null);
	const canvasRef = useRef(null);

	//loads canvas drawing functionality once canvas component is ready
	useEffect(() => {
	  if (canvasRef.current) {
			canvas = new fabric.Canvas(canvasRef.current);
			canvas.isDrawingMode = true;
			canvas.backgroundColor = "black"; canvas.renderAll();
			canvas.freeDrawingBrush.color = "white";
			canvas.freeDrawingBrush.width = BRUSH_SIZE;
	  }
	}, []);
  
  return (
    <>
      <Head>
        <title>AI Digit Recognizer</title>
        <meta name="description" content="An AI-powered handwritten digit recognition app" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className="page">
	  	<div className="main">
      	<h3>AI-Powered Handwritten Digit Recognizer</h3>
	  
				<canvas id="canvas" ref={canvasRef} width={CANV_HEIGHT} height={CANV_WIDTH}></canvas><br/>
	  
	  		<EvaluateBtn />
				<ClearCanvasBtn ref={canvasRef}/>
				<h4 id="result" ref={resultRef}></h4>
      
	  		<p className="paragraph">
	  		Instructions:<br/>
	  		1) Using your cursor or touch screen, draw a digit from 0-9 in the canvas above<br/><br/>
	  		2) Click 'Evaluate' to run the drawing through an artificial neural network (ANN)<br/><br/>
	  		3) The ANN prediction will appear below the canvas<br/><br/>
	  		</p>
		
				<p className="paragraph">
	  		The artificial neural network was built from scratch using Python.
	  		It was trained on the MNIST dataset. This webpage was made using Next.js and 
				is hosted using AWS EC2 with the Python backend route done using Flask. The 
			 	images that you, the user, submit for evaluation are saved to the server using
				SQLite via the sqlite3 Python library. These images will eventually be used for 
				further training of the ANN.<br/><br/>
				- Andrey Skvortsov
	  		</p>
	  	</div>
	  </div>	
	</>
  );
}


function ClearCanvasBtn() {
	const clear = () => {		
		canvas.clear();
		canvas.backgroundColor = "black"; canvas.renderAll();
		console.log("canvas cleared");
	};
	
	return <button onClick={clear}>Clear</button>;
}


function EvaluateBtn() {
	const evaluate = () => {
		const img = processImg();
		const result = runANN(img);
		//saveExample(img, result);
	}

	return <button onClick={evaluate}>Evaluate</button>;
}


async function runANN(image) {
	const response = await fetch('/process', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ 'input_img': image })
	})
	.then(res => {
	    return res.json();
	});

	presentResult(response);
	saveExample(image, response);
	return response;
}


async function saveExample(img, { result }) {
	const response = await fetch('/save', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ 'input_img': img, 'mlp_result': result })
	})
	.then(res => {
		return res.json();
	});
	
	console.log(response.message);
}


function presentResult({ result }) {
	const resultText = resultRef.current;
	resultText.innerText = "Prediction: " + result.toString();
}