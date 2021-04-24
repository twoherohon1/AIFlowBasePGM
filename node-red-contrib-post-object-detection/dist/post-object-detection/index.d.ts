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
    classesURL: string;
    iou: string;
    minScore: string;
    wires: NodeRedWires;
};
declare type NodeRedNodes = {
    createNode(node: any, props: NodeRedProperties): void;
    registerType(type: string, ctor: any): void;
};
declare const _default: (RED: NodeRed) => void;
export = _default;
