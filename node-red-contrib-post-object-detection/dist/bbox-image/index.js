"use strict";
var canvas_1 = require("canvas");
var COLORS = ['Aqua', 'Coral', 'Cyan', 'Yellow', 'GreenYellow'];
var getColor = function colorCounter() {
    var counter = 0;
    return function () {
        counter %= COLORS.length;
        return COLORS[counter++];
    };
}();
module.exports = function init(RED) {
    var BBoxImage = (function () {
        function BBoxImage(config) {
            var _this = this;
            this.maxNumBoxes = 20;
            this.strokeWidth = 2;
            this.fontSize = 10;
            this.objectsProp = 'objects';
            this.imageProp = 'image';
            this.objectsPropType = 'msgPayload';
            this.imagePropType = 'msgPayload';
            if (config.strokeWidth) {
                try {
                    this.strokeWidth = parseInt(config.strokeWidth, 10);
                }
                catch (_a) { }
            }
            if (config.fontSize) {
                try {
                    this.fontSize = parseInt(config.fontSize, 10);
                }
                catch (_b) { }
            }
            if (config.objectsPropType) {
                this.objectsPropType = config.objectsPropType;
                this.objectsProp = config.objectsProp;
            }
            if (config.imagePropType) {
                this.imageProp = config.imageProp;
                this.imagePropType = config.imagePropType;
            }
            RED.nodes.createNode(this, config);
            this.on('input', function (msg) {
                var bboxObjects;
                var bboxImage;
                if (_this.objectsPropType === 'msg') {
                    bboxObjects = msg[_this.objectsProp];
                }
                else {
                    bboxObjects = msg.payload[_this.objectsProp];
                }
                if (_this.imagePropType === 'msg') {
                    bboxImage = msg[_this.imageProp];
                }
                else {
                    bboxImage = msg.payload[_this.imageProp];
                }
                _this.handleRequest(bboxImage, bboxObjects, msg);
            });
            this.on('close', function (done) {
                _this.handleClose(done);
            });
        }
        BBoxImage.prototype.handleRequest = function (image, objects, origMsg) {
            var _this = this;
            if (image === undefined || !Buffer.isBuffer(image) ||
                objects === undefined) {
                this.error('Image object is invalid');
                return;
            }
            var img = new canvas_1.Image();
            img.onload = function () {
                var imgBuff = _this.drawBBox(img, objects);
                var newMsg = origMsg;
                newMsg.payload = imgBuff;
                _this.send(newMsg);
            };
            img.onerror = function (err) {
                _this.error(err.message);
            };
            img.src = image;
        };
        BBoxImage.prototype.handleClose = function (done) {
            done();
        };
        BBoxImage.prototype.drawBBox = function (image, objects) {
            var _this = this;
            var canvas = canvas_1.createCanvas(image.width, image.height);
            var ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0);
            ctx.lineWidth = this.strokeWidth;
            objects.forEach(function (obj) {
                var color = getColor();
                var _a = obj.bbox, x = _a[0], y = _a[1], w = _a[2], h = _a[3];
                if (x < 1) {
                    x = Math.round(x * image.width);
                    w = Math.round(w * image.width);
                    y = Math.round(y * image.height);
                    h = Math.round(h * image.height);
                }
                ctx.font = _this.fontSize + "px sans-serif";
                var txtMet = ctx.measureText(obj.className);
                var ty = y - _this.fontSize - 1;
                ty = ty < 0 ? 0 : ty;
                ctx.strokeStyle = color;
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = color;
                ctx.fillRect(x - 1, ty, txtMet.width + 4, _this.fontSize + 1);
                ctx.fillStyle = 'Black';
                ctx.fillText(obj.className, x + 1, ty + Math.round(txtMet.emHeightAscent));
            });
            return canvas.toBuffer();
        };
        return BBoxImage;
    }());
    RED.nodes.registerType('bbox-image', BBoxImage);
};
//# sourceMappingURL=index.js.map