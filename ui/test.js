const fs = require("fs");
const papa = require("papaparse");
const exec = require('child_process').exec;
const cliProgress = require('cli-progress');

const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);

(async () => {
    const result = await new Promise((resolve) => {
        exec('wc -l ../../data/Ukraine_border.csv', (error, results) => {
            resolve(results);
        });
    })
    bar.start(parseInt(result), 0);
})();

const rows = [];

const parseStream = papa.parse(papa.NODE_STREAM_INPUT, { header: true });

fs.createReadStream("../../data/Ukraine_border.csv", { encoding: "utf-8" })
    .pipe(parseStream)
    .on("data", (row) => {
        rows.push(row)
        bar.update(rows.length);
    })
    .on("end", () => {
        bar.stop();
        console.log(rows.length);
    })
    .on("error", (error) => {
        bar.stop();
        console.log(error.message);
    });

const exportJSONL = (fileName, arr) => {
    const writeStream = fs.createWriteStream(fileName)
    arr.map(x => writeStream.write(`${JSON.stringify(x)}\n`))
    writeStream.end()
    console.log(`exported ${fileName}`);
}

