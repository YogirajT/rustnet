const testString = "aabcccdeeeefffgggggg";

function getRepeatPositions(string) {
    const result = [];

    if (!string?.length) return [];

    let prevElement = null;
    let elementCount = 0;

    for (let i = 0; i < string.length; i++) {
        const element = string[i];
        const nextElement = string[i + 1];

        if (prevElement && prevElement === element) {
            elementCount += 1;

            if (nextElement !== element && elementCount >= 2) {
                result.push([i - elementCount, i]);
            }
        } else {
            elementCount = 0;
        }

        prevElement = element;
    }

    return result;
}

console.log(getRepeatPositions(testString));
