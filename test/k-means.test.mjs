import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { Column, DataTable } from '../src/lib/data-table/index.js';
import { kmeans } from '../src/lib/spatial/k-means.js';
import { logger } from '../src/lib/utils/index.js';

logger.setVerbosity('quiet');

describe('k-means ordered centroid updates', () => {
    it('keeps double-precision sums and ascending source-row order before the f32 store', async () => {
        const points = new DataTable([
            new Column('x', new Float32Array([16777216, 1, -16777216, 3])),
            new Column('y', new Float32Array([1e20, 1, -1e20, 2])),
            new Column('z', new Float32Array([-0, -0, -0, -0]))
        ]);
        const { centroids, labels } = await kmeans(points, 1, 1);
        assert.deepEqual(Array.from(labels), [0, 0, 0, 0]);
        assert.equal(centroids.columns[0].data[0], 1);
        assert.equal(centroids.columns[1].data[0], 0.5);
        assert.equal(Object.is(centroids.columns[2].data[0], 0), true);
    });

    it('retains one random reseed per empty centroid per iteration', async (t) => {
        const random = t.mock.method(Math, 'random', () => 0);
        const points = new DataTable([
            new Column('x', new Float32Array([1, 1, 1, 1])),
            new Column('y', new Float32Array([2, 2, 2, 2]))
        ]);
        const { centroids, labels } = await kmeans(points, 2, 3);
        assert.equal(random.mock.callCount(), 5);
        assert.deepEqual(Array.from(labels), [0, 0, 0, 0]);
        assert.deepEqual(centroids.columns.map(c => Array.from(c.data)), [[1, 1], [2, 2]]);
    });

    it('preserves the small-table clone and typed-label return', async () => {
        const points = new DataTable([new Column('x', new Float32Array([4, 7]))]);
        const result = await kmeans(points, 3, 10);
        assert.ok(result.labels instanceof Uint32Array);
        assert.deepEqual(Array.from(result.labels), [0, 1]);
        assert.deepEqual(result.centroids.columns[0].data, points.columns[0].data);
        assert.notEqual(result.centroids.columns[0].data, points.columns[0].data);
    });
});
