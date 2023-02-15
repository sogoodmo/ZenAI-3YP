import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <div className="video-container">
        <video src="../../../demo_videos_mac/_chair.mp4" autoPlay/>
        {/* <source src="../../../demo_videos_mac/_chair.mp4" type="video/mp4"/> */}
      </div>
    </div>
  );
}

export default App;