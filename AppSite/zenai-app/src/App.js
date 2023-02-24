import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import './App.css';
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000',
});

function App() {
  const [imageReceived, setImageReceived] = useState(false);
  const webcamRef = useRef(null);
  const [capturing, setCapturing] = useState(false);
  const [timeSpent, setTimeSpent] = useState(0);

  useEffect(() => {
    let intervalId = null;

    if (capturing) {
      intervalId = setInterval(() => {
        const curImage = webcamRef.current.getScreenshot();
        
        if (timeSpent === null){
          setTimeSpent(0)
        }
        console.log(timeSpent)
        api.post('/webcam-frame', {data: curImage, timeSpent: timeSpent})
          .then(response => {
            if (response.status === 200 && response.data !== 'error') {
              setImageReceived(response.data.image);
              setTimeSpent(response.data.elapsed_time);
              console.log(response.data)
            } else {
              throw new Error('Failed to send webcam frame');
            }
          })
          .catch(error => {
            console.error(error);
          });
      }, 100);
    } else {
      clearInterval(intervalId);
    }

    return () => clearInterval(intervalId);
  }, [capturing]);

  const startCapture = () => {
    setCapturing(true);
    setTimeSpent(0);
  };

  const stopCapture = () => {
    setCapturing(false);
    setImageReceived(false);
    setTimeSpent(0)
  };
  
  return (
    <div className="App">
      <div className="image-container">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{
            width: 1280,
            height: 720,
            facingMode: 'user',
          }}
        />
      </div>
      <div className="button-container">
        <button onClick={startCapture}>Start capture</button>
        <button onClick={stopCapture}>Stop capture</button>
      </div>
      {imageReceived ? (
        <div>
          <span className="received-message"> Webcam Footage Returned From Server </span>
          <div className="image-received-container">
            <img src={`data:image/jpeg;base64,${imageReceived}`} alt="received" />
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
