// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { imagesNike } from './imagesNike.js'
import { imagesAdidas } from './imagesAdidas.js'

// Number of classes to classify
const NUM_CLASSES = 2;
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;


class Main {
  constructor() {

    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    // this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create training buttons and info texts    
    // for (let i = 0; i < NUM_CLASSES; i++) {
    //   const div = document.createElement('div');
    //   document.body.appendChild(div);
    //   div.style.marginBottom = '10px';

    //   // Create training button
    //   const button = document.createElement('button')
    //   button.innerText = "Train " + i;
    //   div.appendChild(button);

    //   // Listen for mouse events when clicking the button
    //   button.addEventListener('mousedown', () => this.training = i);
    //   button.addEventListener('mouseup', () => this.training = -1);

    //   // Create info text
    //   const infoText = document.createElement('span')
    //   infoText.innerText = " No examples added";
    //   div.appendChild(infoText);
    //   this.infoTexts.push(infoText);
    // }

  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();


    /* imagenes a comparar */
    imagesNike.forEach(src => {
      const img = document.createElement('img');
      img.src = src;
      img.classList.add("training-example-nike");
      img.style.height = '227px';
      img.style.width = '227px';
      document.body.appendChild(img);

      const image = tf.fromPixels(img);

      let logits;
      // // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');
      logits = infer();
      // // Add current image to classifier
      this.knn.addExample(logits, 0)


    });

   imagesAdidas.push("adidasExample2.webp");
   imagesAdidas.push("adidasExample2.webp");
   imagesAdidas.push("adidasExample2.webp");
   imagesAdidas.push("adidasExample2.webp");
   imagesAdidas.push("adidasExample2.webp");
   imagesAdidas.push("adidasExample2.webp");

    imagesAdidas.forEach(src => {
      const img = document.createElement('img');
      img.src = src;
      img.classList.add("training-example-adidas");
      img.style.height = '227px';
      img.style.width = '227px';
      document.body.appendChild(img);

      const image = tf.fromPixels(img);

      let logits;
      // // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');
      logits = infer();
      // // Add current image to classifier
      this.knn.addExample(logits, 1)
    });



    this.start();
  }

  start() {

    // console.log("aca no cargamos las imagenes")
    const exampleCount = this.knn.getClassExampleCount();
    console.log(exampleCount)

    document.getElementById("testButton").addEventListener("click", async   ()=>{
      

      // this.mobilenet = await mobilenetModule.load();
      // this.knn = knnClassifier.create();
    
    const image = document.getElementById("test-case");
    
    
      let logits;
          // // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');
      logits = infer();
      const res = await this.knn.predictClass(logits, TOPK);
    
    console.log(res);

    alert("hay un "+res.confidences['0']*100+" de nike. y un "+res.confidences['1']*100+" de adidas")


    });


  }

  stop() {
    // this.video.pause();
    cancelAnimationFrame(this.timer);
  }

async predict(){
  // const exampleCount = this.knn.getClassExampleCount();
  // console.log(exampleCount)


//   this.mobilenet = await mobilenetModule.load();
//   // this.knn = knnClassifier.create();

// const image = document.getElementById("test-case");


//   let logits;
//       // // 'conv_preds' is the logits activation of MobileNet.
//   const infer = () => this.mobilenet.infer(image, 'conv_preds');
//   logits = infer();
//   const res = await this.knn.predictClass(logits, TOPK);

// console.log(res);

}

  async animate() {
    alert(1)
    if (true) {
      // Get image data from video element
      // const image = tf.fromPixels(this.video);
      // const image = document.getElementsByClassName("training-images-nike")[0];

      // let logits;
      // // 'conv_preds' is the logits activation of MobileNet.
      // const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // // Train class if one of the buttons is held down
      // if (this.training != -1) {
      //   logits = infer();

      //   // Add current image to classifier
      //   this.knn.addExample(logits, this.training)
      // }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
          }
        }
      }

      // Dispose image when done
      // image.dispose();
      // if (logits != null) {
      //   logits.dispose();
      // }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

window.addEventListener('load', () => new Main());
