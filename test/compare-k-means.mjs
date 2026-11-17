import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { Column, DataTable } from '../src/lib/data-table/index.js';
import { kmeans } from '../src/lib/spatial/k-means.js';
import { logger } from '../src/lib/utils/index.js';

const baselinePath = process.argv[2];
if (!baselinePath) throw new Error('usage: node --import tsx test/compare-k-means.mjs /path/to/baseline/src/lib/spatial/k-means.ts');
const baselineUrl = pathToFileURL(resolve(baselinePath));
const baseline = await import(baselineUrl.href);
const baselineLogger = await import(new URL('../utils/logger.ts', baselineUrl).href);
logger.setVerbosity('quiet');
baselineLogger.logger.setVerbosity('quiet');

const generator = seed => () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 2 ** 32;
};
const run = async (implementation, points, clusters, iterations, seed) => {
    const original = Math.random;
    const random = generator(seed);
    const draws = [];
    Math.random = () => {
        const value = random();
        draws.push(value);
        return value;
    };
    try {
        const result = await implementation(points, clusters, iterations);
        return {
            columns: result.centroids.columns.map(c => Buffer.from(c.data.buffer, c.data.byteOffset, c.data.byteLength)),
            labels: Buffer.from(result.labels.buffer, result.labels.byteOffset, result.labels.byteLength),
            draws
        };
    } finally {
        Math.random = original;
    }
};

const results = [];
for (const [rows, dimensions, clusters, repeated] of [[257, 1, 7, false], [1024, 3, 31, false], [2048, 45, 64, false], [37, 45, 12, true], [5, 3, 8, false]]) {
    const values = generator(0x23c7411);
    const points = new DataTable(Array.from({ length: dimensions }, (_, c) => new Column(
        `feature_${c}`, Float32Array.from({ length: rows }, (_, r) => repeated ? (r % 3) * (c + 1) : (values() - 0.5) * 20)
    )));
    for (const seed of [0x19f351, 0x7282f1]) {
        const original = await run(baseline.kmeans, points, clusters, 5, seed);
        const candidate = await run(kmeans, points, clusters, 5, seed);
        assert.deepEqual(candidate.columns, original.columns);
        assert.deepEqual(candidate.labels, original.labels);
        assert.deepEqual(candidate.draws, original.draws);
        results.push({ rows, dimensions, clusters, repeated, seed, randomDraws: candidate.draws.length, bytesEqual: true });
    }
}
console.log(JSON.stringify({ baseline: baselinePath, conditions: 'CPU assignment; identical replayed random streams; five iterations; centroid/label bytes and all random draws compared', results }, null, 2));
