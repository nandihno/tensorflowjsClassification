TFMod = (() => {
    return {
        createTensors() {
            let data = tf.tensor([0, 0, 5, 12, 255, 4, 42, 45], [2, 2, 2]);
            console.log(data.print());

            let values = [];
            for (let i = 0; i < 15; i++) {
                values[i] = Math.random(0, 100) * 100;
            }
            let shape = [3, 5];
            let tense = tf.tensor(values, shape, "int32");
            tense.data().then((results) => {
                console.log(results);
            });
            console.log(tense.dataSync());
            console.log(tense.print());

            let vtense = tf.variable(tense);
            console.log(vtense);


        },
        operations() {
            let values = [];
            for (let i = 0; i < 1500; i++) {
                values[i] = Math.random(0, 100) * 100;
            }
            let shape = [30, 50];
            let tenseA = tf.tensor(values, shape, "int32");
            let tenseB = tf.tensor(values, shape, "int32");
            console.log("using mul");
            let c = tenseA.mul(tenseB);
            console.log(c.print());
            console.log("to use matMul we need to transpose the 2nd tensor");
            let tenseTranspose = tenseB.transpose();
            let d = tenseA.matMul(tenseTranspose);
            console.log(d.print());
            // below is error matMul cols must match no of rows of second element
            // let d = tenseA.matMul(tenseB);
            let values21 = [];
            let values22 = [];
            for (let i = 0; i < 6; i++) {
                values21[i] = Math.random(0, 100) * 100;
                values22[i] = Math.random(0, 100) * 100;
            }
            let tensM1 = tf.tensor(values21, [2, 3], "int32");
            let tensM2 = tf.tensor(values22, [3, 2], "int32");
            let e = tensM1.matMul(tensM2);
            console.log("no need to transpose as tens1 no cols match no of rows of tens2");
            console.log("using matMul", e.print());
        },
        memory() {
            //cleaning using tf.tidy
            for (let i = 0; i < 100; i++) {
                tf.tidy(() => {
                    this.operations();
                });
                console.log(tf.memory().numTensors);
            }
        },
        layers() {
            //this is a model
            let model = tf.sequential();
            //create hidden layer
            let hidden = tf.layers.dense({
                units: 5, //number of nodes
                inputShape: [2], //input shape
                activation: 'sigmoid'
            });
            //add layer
            model.add(hidden);

            let output = tf.layers.dense({
                units: 1, //number of nodes
                activation: 'sigmoid'
            });
            model.add(output);

            //an optimizer using gradient
            let optimizer = tf.train.sgd(0.5);
            let config = {
                optimizer: optimizer,
                loss: tf.losses.meanSquaredError
            };
            model.compile(config);

            //[a,b] ==> 2 inputshapes

            let xs = tf.tensor([
                [0, 0],
                [0.5, 0.5],
                [1, 1]
            ]);

            //[a,b,c] ===> 3 units in the output

            let ys = tf.tensor([
                //[.1,0.5,0.1],
                //[0.5,0,0.2]
                [1],
                [0.5],
                [0]
            ]);


            train().then(() => {
                console.log("complete!");
                let outputs = model.predict(xs);
                //ideally outputs should resemble ys
                outputs.print();
                model.save("localstorage://my-model-1").then((result) => {
                    console.log(result);
                })
            });

            async function train() {
                for (let i = 0; i < 100; i++) {
                    let config = {
                        shuffle: true,
                        epochs: 10
                    };
                    let response = await model.fit(xs, ys, config);
                    console.log(response.history.loss[0], response);
                }
            }
        },
        xor($canvas) {
            let training_data = [{
                inputs: [0, 0],
                outputs: [0]
            },
                {
                    inputs: [0, 1],
                    outputs: [1]
                },
                {
                    inputs: [1, 0],
                    outputs: [1]
                },
                {
                    inputs: [1, 1],
                    outputs: [0]
                }
            ];
            let model = tf.sequential();
            let hidden = tf.layers.dense({
                inputShape: [2],
                units: 2,
                activation: 'sigmoid'
            });
            let output = tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            });
            model.add(hidden);
            model.add(output);

            let optimizer = tf.train.sgd(0.1);
            model.compile({
                optimizer: optimizer,
                loss: tf.losses.meanSquaredError
            });


        }
    }

})();

P5TF = (() => {
    let training_data = [{
        inputs: [0, 0],
        outputs: [0]
    },
        {
            inputs: [0, 1],
            outputs: [1]
        },
        {
            inputs: [1, 0],
            outputs: [1]
        },
        {
            inputs: [1, 1],
            outputs: [0]
        }
    ];
    let resolution = 25;
    let cols;
    let rows;
    let xs;
    let train_xs = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ];
    let train_ys = [
        [0],
        [1],
        [1],
        [0]
    ];


    return {
        getTrainXs() {
            return tf.tensor2d(train_xs);
        },
        getTrainYs() {
            return tf.tensor2d(train_ys);
        },
        training_data: training_data,
        xs: xs,
        resolution: resolution,
        cols: cols,
        rows: rows,
        createModel() {
            let model = tf.sequential();
            let hidden = tf.layers.dense({
                units: 4,
                inputShape: [2],
                activation: 'sigmoid'
            });
            let output = tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            });
            model.add(hidden);
            model.add(output);
            let optimizer = tf.train.sgd(0.1);
            model.compile({
                optimizer: optimizer,
                loss: tf.losses.meanSquaredError
            });
            return model;

        }


    }

})();
let model = null;

function setup() {
    createCanvas(400, 400);
    model = P5TF.createModel();
}

async function trainModel() {
    return await model.fit(P5TF.getTrainXs(),P5TF.getTrainYs());

}

function draw() {
    background(0);
    trainModel().then(result => console.log(result.history.loss[0]));
    noLoop();
    P5TF.resolution = 55;
    P5TF.cols = width / P5TF.resolution;
    P5TF.rows = height / P5TF.resolution;
    let inputs = [];
    for (let i = 0; i < P5TF.cols; i++) {
        for (let j = 0; j < P5TF.rows; j++) {
            let x1 = i / P5TF.cols;
            let x2 = j / P5TF.rows;
            inputs.push([x1, x2]);
        }
    }
    //get predictions
    P5TF.xs = tf.tensor2d(inputs);
    let ys = model.predict(P5TF.xs).dataSync();
    //console.log(ys);

    //draw the results
    let index = 0;
    for (let i = 0; i < P5TF.cols; i++) {
        for (let j = 0; j < P5TF.rows; j++) {
            fill(ys[index] * 255);
            rect(i * P5TF.resolution, j * P5TF.resolution, P5TF.resolution, P5TF.resolution);
            index++;
        }
    }
    /*for (let i=0; i < 10; i++) {
        let data = random(P5TF.training_data);
    }
    */
    //console.log(model);
    //create input data


    //noLoop();


}








