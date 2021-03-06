let trainingData = [

    {
        "input": "my unit test failed",
        "output": "software"
    },
    {
        "input": "tried the program, but it was buggy",
        "output": "software"
    },
    {
        "input": "i need a new power supply",
        "output": "hardware"

    },
    {
        "input": "the drive has a 2TB capacity",
        "output": "hardware"
    },
    {
        "input": "unit-tests",
        "output": "software"
    },
    {
        "input": "program",
        "output": "software"
    },
    {
        "input": "power supply",
        "output": "hardware"
    },
    {
        "input": "drive",
        "output": "hardware"
    },
    {
        "input": "it needs more memory",
        "output": "hardware"
    },
    {
        "input": "code",
        "output": "software"
    },
    {
        "input": "i found some bugs in the code",
        "output": "software"
    },
    {
        "input": "i swapped the memory",
        "output": "software"
    },
    {
        "input": "i tested the code",
        "output": "software"
    },
    {
        "input": "i love computers",
        "output": "hardware"
    },
    {
        "input": "programming",
        "output": "software"
    },
    {
        "input": "i broke my pc",
        "output": "hardware"
    },
    {
        "input": "computer",
        "output": "hardware"
    },
    {
        "input": "buy me a laptop",
        "output": "hardware"
    },
    {
        "input": "buy me a laptop",
        "output": "hardware"
    },
    {
        "input": "sell me an old laptop",
        "output": "hardware"
    },
    {
        "input": "router is not good",
        "output": "hardware"
    },
    {
        "input": "can not connect to phone",
        "output": "hardware"
    },
    {
        "input": "phone",
        "output": "hardware"
    },
    {
        "input": "mobile",
        "output": "hardware"
    },
    {
        "input": "i love to play games",
        "output": "software"
    }

];
let labelList = [
    "software",
    "hardware"
];


let ABCMod = (() => {
    let xInput = [];
    let yInput = [];
    let model = null;
    const MAXSIZE = 60;


    let normaliseData = (charArr) => {
        for (var i = charArr.length; i < MAXSIZE; i++) {
            charArr[i] = " ";
        }
        return charArr;
    };

    let encodeArrString = (text) => {
        let arr = text.split('');
        arr = normaliseData(arr);
        return arr.map(ele => {
            return (ele.charCodeAt(0) / 255);
        });
    };

    let createXTensor = (trainingData) => {
        let array = [];
        let arrInputs = trainingData.map(ele => {
            return ele.input;
        });
        arrInputs.forEach(element => {
            array.push(encodeArrString(element));
        });
        console.log(array);
        return tf.tensor(array);
    };
    let createYTensor = (trainingData) => {
        let arr = trainingData.map(ele => {
            return ele.output;
        });
        let indeces = [];
        arr.forEach((element) => {
            indeces.push(labelList.indexOf(element));
        });
        console.log(arr);
        let labelTensor = tf.tensor1d(indeces, 'int32');
        let ysOneHot = tf.oneHot(labelTensor, 2);
        labelTensor.dispose();
        ysOneHot.print();


        return ysOneHot;
    };

    let createTensorFromTrainingData = (trainingData) => {
        let ob = {
            tensorX: createXTensor(trainingData),
            tensorY: createYTensor(trainingData)
        };
        console.log(ob.tensorX.print());
        return ob;

    };


    return {
        init() {
            model = tf.sequential();
            model.add(tf.layers.dense({
                units: 4,
                inputShape: [MAXSIZE],
                activation: 'sigmoid'
            }));
            model.add(tf.layers.dense({
                units: 2,
                activation: 'softmax'
            }));
            model.compile({
                optimizer: tf.train.sgd(0.6),
                loss: 'categoricalCrossentropy'
            });
        },
        getTensorTrainingObj() {
            return createTensorFromTrainingData(trainingData);
        },
        train(tensorObj) {
            return new Promise((resolve, reject) => {
                $("#trainingMessage").show();
                if (localStorage.getItem("tensorflowjs_models/ABCMod/info") === null) {
                    let config = {
                        epochs: 2000,
                        shuffle: true,
                        callbacks: {
                            onTrainBegin: () => console.log("training started"),
                            onTrainEnd: () => console.log("training has ended!"),
                            onEpochEnd: (num, logs) => {
                                console.log("epoch " + num + " loss is: ", logs);

                            }
                        }
                    };
                    model.fit(tensorObj.tensorX, tensorObj.tensorY, config).then(response => {
                        console.log(response);
                        console.log(response.history.loss[0]);
                        model.save("localstorage://ABCMod").then(result => {
                            console.log("data has been saved", result);
                            $("#trainingMessage").hide();
                            resolve(model);
                        });
                    });
                }
                else {
                    tf.loadModel("localstorage://ABCMod").then(model => {
                        console.log("we have our model!", model);
                        $("#trainingMessage").hide();
                        resolve(model);
                    });
                }
            });
        },
        makePrediction(text) {
            return new Promise((resolve, reject) => {
                let arr = [];
                arr.push(encodeArrString(text));
                let predictTensorX = tf.tensor(arr);
                tf.loadModel("localstorage://ABCMod").then(model => {
                    console.log("we have our model!", model);
                    let output = model.predict(predictTensorX);
                    let indexTensor = output.argMax(1);
                    indexTensor.data().then(result => {
                        console.log(result[0] + " <--- index of the highest one", labelList[result[0]]);
                        let chosenCategory = labelList[result[0]];
                        output.data().then(result => {
                            let retObj = {
                                resultArr: result,
                                category: chosenCategory
                            }
                            console.log("the result is", retObj);
                            //let retArr = result.split(",");
                            resolve(retObj);
                        });
                    });
                });
            });


        }
    }
})();


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
        simpleTraining() {
            let model = tf.sequential();
            model.add(tf.layers.dense({
                units: 1,
                inputShape: [1]
            }));
            //let optimizer = tf.train.sgd(0.5);
            model.compile({
                loss: 'meanSquaredError',
                optimizer: 'sgd'
            });
            let xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
            let ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);
            model.fit(xs, ys, {
                epochs: 450
            }).then(() => {
                console.log("model has been trained!");
                let predictTensor = tf.tensor2d([5], [1, 1]);
                let output = model.predict(predictTensor);
                output.print();
                console.log(output);
                $("#simpleOutput1").text(output.print());

            });
        },
        simpleTraining2(inputs) {
            let model = tf.sequential();
            model.add(tf.layers.dense({
                units: 4,
                inputShape: [2],
                activation: 'sigmoid'
            }));
            model.add(tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            }));
            model.compile({
                optimizer: tf.train.sgd(1),
                loss: 'meanSquaredError'
            });
            //training data
            let xsTrain = tf.tensor([
                [1, 2],
                [3, 2],
                [2, 1],
                [2, 3],
                [1, 3],
                [1, 2]
            ]);
            let ysTrain = tf.tensor([
                [1],
                [1],
                [0],
                [0],
                [1],
                [1]
            ]);
            let predictions = [];
            let predictXs = tf.tensor([
                [3, 1],
                [3, 2],
                [2, 1],
                [2, 3],
                [1, 3],
                [1, 2]
            ]);
            if (localStorage.getItem("tensorflowjs_models/simple2/info") === null) {
                let config = {
                    shuffle: true,
                    epochs: 1000
                };
                model.fit(xsTrain, ysTrain, config).then(response => {
                    console.log(response);
                    let outputs = model.predict(predictXs);
                    outputs.print();
                    model.save("localstorage://simple2").then(result => {
                        console.log("model has been saved", result);
                    });
                    outputs.data().then(result => {
                        $("#simpleOutput2").text(result);
                    })
                });
            }
            else {
                tf.loadModel("localstorage://simple2").then(model => {
                    let outputs = model.predict(predictXs);
                    outputs.print();
                    outputs.data().then(result => {
                        console.log("the result is", result);
                        $("#simpleOutput2").text(result);
                    });
                });
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


$(() => {
    console.log("all loaded");
    $("#createTF").on("click", () => {
        TFMod.createTensors();
    });
    $("#operations").on("click", () => {
        TFMod.operations();
    });
    $("#memory").on("click", () => {
        TFMod.memory();
    });
    $("#layers").on("click", () => {
        TFMod.layers();
    });
    $("#xor").on("click", () => {
        TFMod.xor($("#myCanvas"));
    });
    $("#simple1").on("click", () => {
        TFMod.simpleTraining();
    });
    $("#simple2").on("click", () => {
        $("#t1").val();
        TFMod.simpleTraining2();

    });
    $("#clearStorage").on("click", () => {
        localStorage.clear();
    });

    $("#testTrain").on("click", () => {
        ABCMod.init();
        let tensorObj = ABCMod.getTensorTrainingObj();
        ABCMod.train(tensorObj);
    });
    $("#dataTestBtn").on("click", () => {
        let text = $("#dataTest").val();
        console.log("our text is", text);
        ABCMod.makePrediction(text).then(retObj => {
            let arr = retObj.resultArr;
            let category = retObj.category;
            console.log(arr, arr[0], arr[1]);
            $("#foodPC").text(arr[0]);
            $("#carsPC").text(arr[1]);
            $("#categorySpan").text(category);
        });
    });
});


