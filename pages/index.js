import Head from "next/head";
import Image from "next/image";
import styles from "@/styles/Home.module.css";

export default function Home() {
  return (
    <>
      <Head>
        <title>AI Digit Recognizer</title>
        <meta name="description" content="An AI-powered handwritten digit recognition app" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className="main">
      	<h3>AI-Powered Handwritten Digit Recognizer</h3>
	  
		<div id="canvas">here goes the drawing canvas</div>
	  
	  	<button>Evaluate</button>
	  	<button>Clear</button>
      
	  	<p>Here I will put the description of the website and its features 
	  		yada yada yada yada yada yada <br/>yada. Need to constrain the size of this paragraph
	  		horizontally for the sake of mobile
	  	</p>
	  
	  </div>
    
	
	
	
	</>
  );
}
