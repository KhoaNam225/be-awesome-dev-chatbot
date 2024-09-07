---
title: Javascript Modern Syntax
author: Nam Pham
dateCreated: 02-02-2024
dateModified: 02-02-2024
abstract: ES6 is a version of Javascript which was release in 2015. It introduces a lot of new syntax that improve the productivity of the developer. This section presents some important syntax that a developer should know to be use this language effectively.
tags:
  - Javascript Basics
  - Web Development
---

## The Spread Operator

**Syntax:** `...[variable_name]`

**Usage:** The spread operator is used to expand an iterable (a string, object or an array). It provides a handy way to "spread" the content of that iterable to be used in a list of some kind.

The following code demonstrates the spread operator's usage:

```typescript
const originalArray = [1, 2, 3, 4]

const copiedArray = [...originalArray]

console.log(originalArray) // This will give: [1, 2, 3, 4]
console.log(copiedArray) // This will also give: [1, 2, 3, 4]

/* Spread operator also works well with objects */
const originalObject = {
  year: 2022,
  month: 1,
}

const copiedObject = { ...originalObject }

console.log(originalObject) // This will give: { year: 2022, month: 1 }
console.log(copiedObject) // This will also give: { year: 2022, month: 1 }
```

**Use Case - Shallow Copy:** The spread operator can be used to conveniently create shallow copies of the contents of arrays and objects. Specifically, the new array/object will have new copies (new instances) of any [primitives](https://developer.mozilla.org/en-US/docs/Glossary/Primitive) values (numbers, strings, booleans, null/undefined etc.). For non-primitive values (e.g. other objects/arrays nested in the original) in the shallow copy, only the reference is used (i.e. objects and arrays are shared between the clone and original).

For example, consider the below script:

```typescript
const originalArray = [1, 'hello', true, [5, 6, 7], { prop: 100 }]
const clonedArray = [...originalArray]

console.log(clonedArray) // This will give [1, "hello", true, [5, 6, 7], {prop: 100}]

originalArray[0] = 7
console.log(originalArray[0], clonedArray[0]) // This will give 7 1

originalArray[4].prop = 200
console.log(originalArray[4], clonedArray[4]) // This will give {prop: 200} {prop: 200}
```

Note that we can also produce a shallow clone using `clonedArray = originalArray.concat([])` or with`clonedArray = originalArray.slice()`

**Use Case - Populating Array and Object Literals:** The above shallow-copying behaviour can be extended to quickly and easily build new array literals by spreading existing ones. The order of elements inside the spread array, and the order of those elements relative to other elements in the array are preserved:

```typescript
const template = [3, 4, 5]

const myLiteral1 = [1, 2, ...template] // [1, 2, 3, 4, 5]
const myLiteral2 = [2, ...template, 6] // [2, 3, 4, 5, 6]
const myLiteral3 = [...template, 6, 7] // [3, 4, 5, 6, 7]

// If spreading is not used
const myLiteral4 = [2, template, 6] // [2, [3, 4, 5], 6]
```

Similarly, this works for object literals. However as the same key can be used multiple times in the same object literal (because of using the spread operator), only the last key-value pair is used for each key.

```typescript
const template1 = { a: 1, b: 2, c: 3 }
const template2 = { b: 20, c: 30, d: 40 }

const myLiteral1 = { a: 100, ...template1 } // {a: 1, b: 2, c: 3}
const myLiteral2 = { a: 100, ...template1, ...template2, c: 300 } // {a: 1, b: 20, c: 300, d: 40}
const myLiteral3 = { ...template2, c: 300, d: 400 } // {b: 20, c: 300, d: 400}
```

**Use Case - Spreading arguments:** The same syntax can also be used when specifying ordered arguments to a function or method.
Specifically, spreading an array will specify arguments in that order. Note that spreading objects to specify arguments by key does NOT work (this is usually solved by passing in a single object with all required arguments instead).

```typescript
function concatThreeStrings(arg1, arg2, arg3) {
  return arg1 + arg2 + arg3
}

const myArgs1 = ['a', 'b', 'c']
concatThreeStrings(...myArgs) // "abc"

const myArgs2 = ['a', 'b']
concatThreeStrings(...myArgs, 'c') // "abc"

const myArgs3 = { arg3: 'c', arg1: 'a', arg2: 'b' }

// NOTE: This does NOT work!
concatThreeStrings(...myArgs3) // Uncaught TypeError: Found non-callable @@iterator

function concatThreeStringsFromObject(input) {
  const { arg1, arg2, arg3 } = input
  return arg1 + arg2 + arg3
}

concatThreeStringsFromObject(myArgs3) // "abc"
```

**Use Case - Collating arguments:** A similar syntax can also be used when defining functions with an unspecified number of arguments. With this syntax, all arguments are collated into a single array for use within the function.

```typescript
function sum(...arguments) {
  return arguments.reduce((acc, value) => acc + value)
}

sum(10, 20, 30, 40) // 100
```

## The Destructuring Operator

**Syntax:** `const [var1, var2, var3, ..., varN] = array` for arrays and `const { var1, var2, var3, ..., varN } = object` for objects.

**Usage:** The destructuring operator is quite opposite to the spread operator. Instead of expanding the whole content of an array/object, it will 'pick' some values in that array/object and assign them to individual variables. For example:

```typescript
const arrayA = [1, 2, 3]
const [first, second, last] = arrayA

console.log(first) // 1
console.log(second) // 2
console.log(last) // 3

const objA = {
  year: 2021,
  month: 1,
  day: 31,
}
const { year, month, day } = objA

console.log(year) // 2021
console.log(month) // 1
console.log(day) // 31
```

When working with arrays, the destructuring operator will be useful when we want to pick the values at some certain positions that we already know beforehand. For example:

```typescript
const arrayA = [1, 2, 3, 4, 5]
const [one, two, ...rest] = arrayA

console.log(one) // 1
console.log(two) // 2
console.log(rest) // [3, 4, 5]

// The above is equivalent to the following more 'traditional' way
const one = arrayA[0]
const two = arrayA[1]
const rest = arrayA.slice(2)

// Also, with the destructuring operator, we can easily swap values of 2 variables like this:
let a = 10
let b = ((20)[(a, b)] = [b, a]) // Now a = 20 and b = 10!

console.log(a) // 20
console.log(b) // 10

// The above is equivalent to this more usual way
let temp = a
a = b
b = temp
```

You might not see the real power of the destructuring operator from the example with array above since it is more widely used with objects. This operator will give us an incredibly quick way to access the values of individual properties of an object. The below example demonstrates how destructuring operator is used to avoid writing tedious and duplicated code:

```typescript
  const originalObj = {
    year: 2021,
    month: 1,
    day: 31,
    tonnes: 1000,
    asset: 'AC',
    product: 'F',
    destinationAsset: 'MAC_SY1'
  }

  const asset = originalObj.asset
  const product = originalObj.product
  //.... blah... blah... Too long, didn't write!


  const {
    year,
    month,
    day,
    asset,
    tonnes,
    product
    destinationAsset
  } = originalObj

  console.log(year)  // 2021
  console.log(month)  // 1
  // ... other variables will have the corresponding values
```

Also, destructuring operator gives us a way to selectively choose/remove a property from a given object. For example, in the below script, we select the value of the `year` property and assign the rest to the `theRest` object. In other words, we have removed the `year` property from the original object!

```typescript
  const obj = {
    year; 2021,
    month: 1,
    day: 31,
    tonnes: 1000
  }

  // Only get the value of year and assign the rest to an object
  const { year, ...theRest } = obj

  console.log(year)  // 2021
  console.log(theRest)  // { month: 1, day: 31, tonnes: 1000 }, year has now been removed!


  // Get the value of tonnes and assign it to another variable called 'ofrTonnes'
  const {
    tonnes: ofrTonnes,
    year,
    month,
    day } = obj

  console.log(ofrTonnes)  // 1000
```

Finally, we can also leverage the destructuring operator to shorten the amount of code we need to write to create a function. For example, the below function takes in an object of type `TonnageValue` and performs some calculation with it. There are 3 different ways to implement this function:

```typescript
interface TonnageValue {
  year: number
  month: number
  asset: string
  tonnes: number
}

// The 'traditional' way of writing a function that accept a TonnageValue object
function processTonnageValue(tonnageVal: TonnageValue) {
  const year = tonnageVal.year
  const month = tonnageVal.month
  const asset = tonnageVal.asset
  const tonnes = tonnageVal.tonnes

  // ... Do other calculations
}

// That same function but improved a bit
function processTonnageValueImproved(tonnageVal: TonnageValue) {
  const { year, month, asset, tonnes } = tonnageVal

  // ... Do other calculations
}

// Let's bring it to the ultimate form!
function processTonnageValueImprovedVer2({
  year,
  month,
  asset,
  tonnes,
}: TonnageValue) {
  // ... Do other calculations
}
```

## Shorthand Syntax

Apart from the spread and destructuring operators, Javascript ES6+ also provide some more useful syntax that can help reduce the amount of code that we have to write and make it a lot more readable.

### 1. The Ternary Operator

**Syntax:** `[conditional_statement] ? [value_if_true] : [value_if_false]`

**Usage:** The ternary operator is used to return a value base on a particular condition. Its main purpose is to replace simple `if...else...` statements and thus reduce the amount of code written. For example:

```typescript
const age = 26

// This is the long way of assigning proper value to beverage by checking
// for age
let beverage = ''
if (age < 18) {
  beverage = 'juice'
} else {
  beverage = 'beer'
}

// Using ternary operator is a lot faster
let beverage = age < 18 ? 'juice' : 'beer'
```

### 2. The "OR" operator

**Syntax:** `[variable_x] || [value_to_return_if_x_is_falsy]`

In some situations, we might need to check if a certain variable has falsy value (`null, NaN, 0, '', undefined`) to decide if we should use it or another default value.

In this case, we can use the ternary operator mentioned in the previous section to shorten the code. However, it can be shortened even more if we use the OR operator. Like this:

```typescript
const userA = {
  username: 'Peter Parker',
  age: 26,
}

const userB = {
  username: '',
  age: 20,
}

// Use ternary operator
const nameA = userA.username ? userA.username : 'Anonymous'
const nameB = userB.username ? userB.username : 'Anonymous'

// Or we can use OR operator
const nameA = userA.username || 'Anonymous'
const nameB = userB.username || 'Anonymous'

console.log(nameA) // 'Peter Parker'
console.log(nameB) // 'Anonymous' because empty string is falsy
```

### 3. The Null-ish Coalescing Operator

One drawback of the logical OR operator is that it will return the value to the right of the operator whenever the value to the left is falsy. However, sometimes we might consider `''` or `0` as accepted values (e.g. tonnage of 0 is still a valid number). In those situation, the null-ish coalescing syntax will be useful. It will only return the value to the right if the left value is either `null` or `undefined`. For example:

```typescript
const userA = {
  username: '',
  age: 26,
}

const userB = {
  age: 20,
}

const nameA = userA.username ?? 'Anonymous'
const nameB = userB.username ?? 'Anonymous'

console.log(nameA) // ''
console.log(nameB) // 'Anonymous' since username is undefined in userB
```

### 4. The Optional Chaining Operator

When working with an object, sometimes we will need to check if it has a certain property or method. If yes, we proceed with using that property or method, otherwise we do something else. This can be easily done with an `if... else...` statement to first check for the existence of the property/method and then perform some actions accordingly:

```typescript
const john = {
  name: 'John',
  cat: {
    name: 'Tom',
    meow: () => {
      console.log('Meow!')
    },
  },
}

const mary = {
  name: 'Mary',
  dog: {
    name: 'Chloe',
    bark: () => {
      console.log('Goof!')
    },
  },
}

// Since mary may or may not have a dog
// we must check if she rally has a dog before letting him/her bark
if (mary.dog) {
  if (mary.dog.bark) {
    mary.dog.bark()
  }
}
```

However, if the object is deeply nested, using `if...else...` statements will result in a very big chunk of code even if we only want to perform some simple action:

```typescript
const mary = {
  name: 'Mary',
  dog: {
    name: 'Chloe',
    puppy: {
      bark: () => {
        console.log('Goof!)
      }
    }
  }
}

// We check if Mary's dog has a puppy
// if yes, we let it bark
if (mary.dog) {
  if (mary.dog.puppy) {
    if (mary.dog.puppy.bark) {
      mary.dog.puppy.bark()
    }
  }
}
```

To tackle this problem , we can use the optional chaining syntax. This syntax will first check if a property/method exist. If yes, it will continue down the chain, otherwise it will return `undefined` immediately without throwing any error. For example:

```typescript
const mary = {
  name: 'Mary',
  dog: {
    name: 'Chloe',
    puppy: {
      bark: () => {
        console.log('Goof!)
      }
    }
  }
}

// This will print to the console if Mary really has a dog, which also has a
mary.dog?.puppy?.bark?()
```

## The Arrow Function

In addition to the regular way of defining functions using the `function` keyword, functions can also be defined using **Arrow Syntax**. Arrow functions are particularly convenient when defining **callback** functions, for example to use in Array methods.

**Syntax:** `(arg1, arg2, ...) => { return myResult }`

The code below shows some of the syntax that can be used when defining arrow functions.

```typescript
// A 'do nothing' function
const function1 = () => {}

// With some arguments
const function2 = (arg1, arg2) => {}

// If only one line is required and result is not an object literal, can omit braces
const function3 = (arg1, arg2) => arg1 + arg2

// If only one line is required and result is an object literal, 'escape' braces with parentheses
const function4 = (arg1, arg2) => ({ sum: arg1 + arg2 })

// If multiple lines are required, use 'return' to return the result
const function5 = (arg1, arg2) => {
  return { sum: arg1 + arg2 }
}

// You can still destructure arguments
const function6 = ([arg1, arg2]) => {
  return \`\${arg1}\${arg2}\`
}

const myInput = { a: 1, b: 2, c: 3 }
const myEntries = Object.entries(myInput) // [[a, 1], [b, 2], [c, 3]]
myEntries.map(function6)                  // ["a1", "b2", "c3"]
```

Please note that arrow functions are not the same as functions defined with the `function` keyword. Some differences exist, such as the behaviour of the `this` keyword, and the inability to define recursive arrow functions.
