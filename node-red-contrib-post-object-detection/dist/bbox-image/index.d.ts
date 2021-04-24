declare type NodeRed = {
    nodes: NodeRedNodes;
};
declare type NodeRedWire = {
    [index: number]: string;
};
declare type NodeRedWires = {
    [index: number]: NodeRedWire;
};
declare type NodeRedProperties = {
    id: string;
    type: string;
    name: string;
    wires: NodeRedWires;
    strokeWidth: string;
    fontSize: string;
    objectsProp: string;
    objectsPropType: string;
    imageProp: string;
    imagePropType: string;
};
declare type NodeRedNodes = {
    createNode(node: any, props: NodeRedProperties): void;
    registerType(type: string, ctor: any): void;
};
declare const _default: (RED: NodeRed) => void;
export = _default;
