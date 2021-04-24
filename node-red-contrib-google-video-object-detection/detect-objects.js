module.exports = function (RED) {
  const NODE_TYPE = "google-video-object-detection";
  const video = require('@google-cloud/video-intelligence');
  const fs = require('fs');
  const util = require('util');

  function DetectorNode(config) {
    
    // Set up node
    RED.nodes.createNode(this, config);
    const node = this;

    // Get Google API credentials
    const keyFilename = config.keyFilename;

    // Set up Google Video API client
    const client = new video.VideoIntelligenceServiceClient({
      "keyFilename": keyFilename
    });

    /**
     * Input function
     * 
     * We are expecting either:
     * > msg.filename (image source) or...
     * > msg.payload (image base64 data)
     */
    async function Input(msg) {

      let base64 = null;

      // Check for message filename
      if (msg.filename) {
        // If we have a filename we need to grab the file
        const readFile = util.promisify(fs.readFile);
        const file = await readFile(filename);
        base64 = file.toString('base64');
      } else {
        // Check if we've been given a buffer
        if (msg.payload) {
          base64 = msg.payload;
        } else {
          // No msg.filename and msg.payload is not a buffer.
          node.error("Neither msg.filename nor msg.payload properly supplied.");
          return;
        }
      }

      // Create request object
      const request = {
        inputContent: base64,
        features: ['OBJECT_TRACKING'],
        locationId: 'us-east1'
      };

      // Make the request
      try {
        const [operation] = await client.annotateVideo(request);
        const [operationResult] = await operation.promise();
        const annotations = operationResult.annotationResults[0];

        msg.payload = annotations;
        node.send(msg);
      } catch(err) {
        node.error(err);
      }
    }

    // Handler for input
    node.on("input", Input);

  }

  RED.nodes.registerType(NODE_TYPE, DetectorNode);
};