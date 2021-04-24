/**
 * Copyright 2016 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Authors:
 *	- Olaf Hahn
 **/


module.exports = function(RED) {
	"use strict";

	var settings = RED.settings;
	var events = require("events");
	var exec = require("child_process").exec;
	var isUtf8 = require("is-utf8");
	var bufMaxSize = 32768;  // Max serial buffer size, for inputs...


	// CameraPI Take Photo Node
	function CameraPiTakePhotoNode(config) {
		// Create this node
		RED.nodes.createNode(this,config);

		// set parameters and save locally
		this.filemode = config.filemode;
		this.filename =  config.filename;
		this.filedefpath = config.filedefpath;
		this.filepath = config.filepath;
		this.fileformat = config.fileformat;
		this.resolution =  config.resolution;
		this.rotation = config.rotation;
		this.fliph = config.fliph;
		this.flipv = config.flipv;
		this.sharpness = config.sharpness;
		this.brightness = config.brightness;
		this.contrast = config.contrast;
		this.imageeffect = config.imageeffect;
		this.exposuremode = config.exposuremode;
		this.iso = config.iso;
		this.agcwait = config.agcwait;
		this.quality = config.quality;
		this.led = config.led;
		this.awb = config.awb;
		this.name =  config.name;
		this.activeProcesses = {};

		var node = this;

		// if there is an new input
		node.on("input", function(msg) {

			var fsextra = require("fs-extra");
			var fs = require("fs");
			var uuidv4 = require("uuid/v4");
			var uuid = uuidv4();
			var os = require("os");
			var localdir = __dirname;
			var homedir = os.homedir();
			var defdir = "D:/hero/pic.jpg";
			var cl = "python " + localdir + "/lib/python/get_photo.py";
			var resolution;
			var fileformat;
			var filename;
			var filepath;
			var filemode;
			var filefqn;
			var fliph, flipv;
			var sharpness;
			var brightness;
			var contrast;
			var imageeffect;
			var agcwait;
			var quality;
			var led;
			var awb;
			var rotation;
			var exposuremode;
			var iso;

			node.status({fill:"green",shape:"dot",text:"connected"});

			// Check the given filemode
			if((msg.filemode) && (msg.filemode !== "")) {
				filemode = msg.filemode;
			} else {
				if (node.filemode) {
					filemode = node.filemode;
				} else {
					filemode = "1";
				}
			}

			if (filemode == "0") {
				// Buffered mode (old Buffermode)
				filename = "pic_" + uuid + ".jpg";
				fileformat = "jpeg";
				filepath = homedir + "/";
				filefqn = filepath + filename;
				if (RED.settings.verbose) { node.log("camerapi takephoto:" + filefqn); }
				console.log("CameraPi (log): Tempfile - " + filefqn);

				cl += " " + filename + " " + filepath + " " + fileformat;
			} else if (filemode == "2") {
				// Auto file name mode (old Generate)
				filename = "pic_" + uuid + ".jpg";
				fileformat = "jpeg";
				filepath = defdir;
				filefqn = filepath + filename;
				if (RED.settings.verbose) { node.log("camerapi takephoto:" + filefqn); }
				console.log("CameraPi (log): Generate - " + filefqn);

				cl += " " + filename + " " + filepath + " " + fileformat;
			} else {
				 // Specific FileName
				 if ((msg.filename) && (msg.filename.trim() !== "")) {
						filename = msg.filename;
				} else {
					if (node.filename) {
						filename = node.filename;
					} else {
						filename = "pic_" + uuid + ".jpg";
					}
				}
				cl += " " + filename;

				if (node.filedefpath == "1" ) {
					filepath = defdir;
				} else {
					if ((msg.filepath) && (msg.filepath.trim() !== "")) {
						filepath = msg.filepath;
					} else {
						if (node.filepath) {
							filepath = node.filepath;
						} else {
							filepath = defdir;
						}
					}
				}
				cl += " " + filepath;

				if ((msg.fileformat) && (msg.fileformat.trim() !== "")) {
					fileformat = msg.fileformat;
				} else {
					if (node.fileformat) {
						fileformat = node.fileformat;
					} else {
						fileformat = "jpeg";
					}
				}
				cl += " " + fileformat;
				if (RED.settings.verbose) { node.log("camerapi takephoto:" + filefqn); }
			}
			if (RED.settings.verbose) { node.log(cl); }

			filefqn = filepath + filename;

			var child = exec(cl, {encoding: "binary", maxBuffer:10000000}, function (error, stdout, stderr) {
				var retval = new Buffer(stdout,"binary");
				try {
					if (isUtf8(retval)) { retval = retval.toString(); }
				} catch(e) {
					node.log(RED._("exec.badstdout"));
				}

				// check error
				var msg2 = {payload:stderr};
				var msg3 = null;
				//console.log("[exec] stdout: " + stdout);
				//console.log("[exec] stderr: " + stderr);
				if (error !== null) {
					msg3 = {payload:error};
					console.error("CameraPi (err): " + error);
					msg.payload = "";
					msg.filename = "";
					msg.fileformat = "";
					msg.filepath = "";
				} else {
					msg.filename = filename;
					msg.filepath = filepath;
					msg.fileformat = fileformat;

					// get the raw image into payload and delete tempfile on buffermode
					if (filemode == "0") {
						// put the imagefile into payload
						msg.payload = fs.readFileSync(filefqn);

						// delete tempfile
						fsextra.remove(filefqn, function(err) {
						  if (err) return console.error("CameraPi (err): " + err);
						  console.log("CameraPi (log): " + filefqn + " remove success!")
						});			   
					} else {
						msg.payload = filefqn;
						console.log("CameraPi (log): " + filefqn + " written with success!")
					}
				}

				node.status({});
				node.send(msg);
				delete node.activeProcesses[child.pid];
			});

			child.on("error",function(){});

			node.activeProcesses[child.pid] = child;

		});

		// CameraPi-TakePhoto has a close
		// New function signature function(removed, done) included in Node-Red 0.17
		node.on("close", function(removed, done) {
			if (removed) {
				// This node has been deleted
				node.closing = true;
			}
			else {
				// This node is being restarted
			}
			done();
		});
	}
	RED.nodes.registerType("camerapi-takephoto",CameraPiTakePhotoNode);
}
