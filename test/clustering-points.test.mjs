import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { Column, DataTable } from '../src/lib/data-table/index.js';
import { ClusteringPoints } from '../src/lib/gpu/clustering-points.js';

const deviceStandIn = () => ({
    buffers: new Set(),
    _vram: { sb: 0 },
    allocations: [],
    writes: [],
    createBufferImpl() {
        const device = this;
        const implementation = {
            destroyed: false,
            allocate(_device, size) {
                this.bytes = new Uint8Array(size);
                device.allocations.push(size);
            },
            write(_device, offset, data, dataOffset, size) {
                assert.equal(this.destroyed, false);
                const values = data.subarray(dataOffset, dataOffset + size);
                this.bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), offset);
                device.writes.push({ offset, bytes: values.byteLength });
            },
            destroy() {
                assert.equal(this.destroyed, false);
                this.destroyed = true;
            }
        };
        return implementation;
    }
});

const points = n => new DataTable([
    new Column('x', Float32Array.from({ length: n }, (_, i) => 2 ** i)),
    new Column('y', Float32Array.from({ length: n }, (_, i) => -(2 ** i))),
    new Column('z', new Float32Array(n).fill(0.5))
]);

const expectBatch = (buffer, batch, rows, useF16) => {
    const values = useF16 ? new Uint16Array(buffer.impl.bytes.buffer) : new Float32Array(buffer.impl.bytes.buffer);
    for (let i = 0; i < rows; i++) {
        const exponent = batch * 3 + i;
        const positive = useF16 ? (15 + exponent) << 10 : 2 ** exponent;
        const negative = useF16 ? positive | 0x8000 : -(2 ** exponent);
        assert.deepEqual(Array.from(values.subarray(i * 3, i * 3 + 3)), [positive, negative, useF16 ? 0x3800 : 0.5]);
    }
};

describe('immutable clustering point buffers', () => {
    for (const useF16 of [false, true]) {
        it(`uploads each of four batches once and preserves ${useF16 ? 'f16' : 'f32'} packed values`, () => {
            const device = deviceStandIn();
            const cache = new ClusteringPoints(device, points(11), 3, useF16);
            for (let pass = 0; pass < 10; pass++) {
                for (let batch = 0; batch < 4; batch++) {
                    expectBatch(cache.get(batch), batch, batch === 3 ? 2 : 3, useF16);
                }
            }
            assert.equal(device.writes.length, 4);
            assert.deepEqual(device.allocations, new Array(4).fill(useF16 ? 20 : 36));
            assert.equal(device.buffers.size, 4);
            cache.destroy();
            assert.equal(device.buffers.size, 0);
            assert.equal(device._vram.sb, 0);
        });
    }

    it('streams overflow batches through the fourth buffer without stale values or unbounded allocations', () => {
        const device = deviceStandIn();
        const cache = new ClusteringPoints(device, points(17), 3, false);
        for (let pass = 0; pass < 3; pass++) {
            for (let batch = 0; batch < 6; batch++) {
                expectBatch(cache.get(batch), batch, batch === 5 ? 2 : 3, false);
            }
        }
        assert.equal(device.writes.length, 12);
        assert.equal(device.allocations.length, 4);
        cache.destroy();
        assert.equal(device._vram.sb, 0);
    });
});
