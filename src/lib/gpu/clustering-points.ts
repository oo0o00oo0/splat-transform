import { BUFFERUSAGE_COPY_DST, FloatPacking, GraphicsDevice, StorageBuffer } from 'playcanvas';

import { DataTable } from '../data-table';

const interleaveData = (result: Uint16Array | Float32Array, dataTable: DataTable, numRows: number, rowOffset: number) => {
    const { numColumns } = dataTable;
    if (result instanceof Uint16Array) {
        for (let c = 0; c < numColumns; ++c) {
            const column = dataTable.columns[c];
            for (let r = 0; r < numRows; ++r) {
                result[r * numColumns + c] = FloatPacking.float2Half(column.data[rowOffset + r]);
            }
        }
    } else {
        for (let c = 0; c < numColumns; ++c) {
            const column = dataTable.columns[c];
            for (let r = 0; r < numRows; ++r) {
                result[r * numColumns + c] = column.data[rowOffset + r];
            }
        }
    }
};

class ClusteringPoints {
    private device: GraphicsDevice;
    private points: DataTable;
    private batchSize: number;
    private interleaved: Uint16Array | Float32Array;
    private buffers: StorageBuffer[] = [];
    private batches: number[] = [];

    constructor(device: GraphicsDevice, points: DataTable, batchSize: number, useF16: boolean) {
        this.device = device;
        this.points = points;
        this.batchSize = batchSize;
        const values = points.numColumns * batchSize;
        this.interleaved = useF16 ? new Uint16Array(Math.ceil(values / 2) * 2) : new Float32Array(values);
    }

    get(batch: number): StorageBuffer {
        // Three retained batches plus one streaming slot bound even oversized environment tables.
        const slot = Math.min(batch, 3);
        if (!this.buffers[slot]) {
            this.buffers[slot] = new StorageBuffer(this.device, this.interleaved.byteLength, BUFFERUSAGE_COPY_DST);
        }
        const buffer = this.buffers[slot];
        if (this.batches[slot] !== batch) {
            const numRows = Math.min(this.points.numRows - batch * this.batchSize, this.batchSize);
            interleaveData(this.interleaved, this.points, numRows, batch * this.batchSize);
            const values = this.points.numColumns * numRows;
            buffer.write(0, this.interleaved, 0, this.interleaved instanceof Uint16Array ? Math.ceil(values / 2) * 2 : values);
            this.batches[slot] = batch;
        }
        return buffer;
    }

    destroy() {
        for (const buffer of this.buffers) {
            buffer?.destroy();
        }
        this.buffers = [];
        this.batches = [];
    }
}

export { ClusteringPoints, interleaveData };
