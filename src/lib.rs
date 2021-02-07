#![feature(maybe_uninit_extra)]
#![feature(array_methods)]
use num::traits::{One, Zero};
use std::convert::From;
use std::iter::Sum;
use std::mem::{self, MaybeUninit};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::slice::{Iter, IterMut};
use thiserror::Error;

#[derive(Debug, PartialEq, Eq, Clone)]
///C is the type of the coefficients of the Matrix
///ROWS is the number of lines & COLS the number of columns
pub struct Matrix<C, const ROWS: usize, const COLS: usize> {
    data: [[C; COLS]; ROWS],
}

impl<C, const ROWS: usize, const COLS: usize> Matrix<C, ROWS, COLS> {
    ///Returns an iterator of all lines of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    ///# use matrix::Matrix;
    /// let mat = Matrix::from([[9, 8, 7], [6, 5, 4]]);
    /// assert_eq!(mat.get_lines().nth(1), Some(&[6, 5, 4]));
    /// ```
    pub fn get_lines(&self) -> Iter<[C; COLS]> {
        self.data.iter()
    }

    ///Returns a mutable iterator of all lines of the matrix.
    /// See [`get_lines`] for examples.
    ///
    /// [`get_lines`]: #method.get_lines
    pub fn get_mut_lines(&mut self) -> IterMut<[C; COLS]> {
        self.data.iter_mut()
    }

    ///Returns a reference to a line or `None` if index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    ///# use matrix::Matrix;
    /// let mat = Matrix::from([[9, 8, 7], [6, 5, 4]]);
    /// assert_eq!(mat.get_line(1), Some(&[6, 5, 4]));
    /// ```
    pub fn get_line(&self, index: usize) -> Option<&[C; COLS]> {
        self.data.get(index)
    }

    ///Returns a mutable reference to a line or `None` if index is out of bounds.
    /// See [`get_line`] for examples.
    ///
    /// [`get_line`]: #method.get_line
    pub fn get_mut_line(&mut self, index: usize) -> Option<&mut [C; COLS]> {
        self.data.get_mut(index)
    }

    ///Returns a reference to a single coefficient or `None` if either `row` or `col` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    ///# use matrix::Matrix;
    /// let mat = Matrix::from([[9, 8, 7], [6, 5, 4]]);
    /// assert_eq!(mat.get(1, 2), Some(&4));
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<&C> {
        match self.data.get(row) {
            None => None,
            Some(line) => line.get(col),
        }
    }

    ///Returns a mutable reference to a single coefficient or `None` if either `row` or `col` is out of bounds.
    /// See [`get`] for examples.
    ///
    /// [`get`]: #method.get
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut C> {
        match self.data.get_mut(row) {
            None => None,
            Some(line) => line.get_mut(col),
        }
    }
}

impl<C, const ROWS: usize, const COLS: usize> From<[[C; COLS]; ROWS]> for Matrix<C, ROWS, COLS> {
    fn from(data: [[C; COLS]; ROWS]) -> Self {
        Matrix { data }
    }
}

///Multiplication by a coefficient. Can never fail, works matrices of all dimensions.
///Similar to the `dilate` method of square matrices but for all lines at once.
impl<'a, C: 'a, const ROWS: usize, const COLS: usize> MulAssign<&'a C> for Matrix<C, ROWS, COLS>
where
    C: MulAssign<&'a C> + Copy,
{
    fn mul_assign(&mut self, coef: &'a C) {
        for row in self.data.iter_mut() {
            for c in row.iter_mut() {
                *c *= coef
            }
        }
    }
}

///Matrix addition, they must be of the same size
impl<C, const ROWS: usize, const COLS: usize> AddAssign<Matrix<C, ROWS, COLS>>
    for Matrix<C, ROWS, COLS>
where
    C: AddAssign + Copy,
{
    fn add_assign(&mut self, other: Matrix<C, ROWS, COLS>) {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(row_a, row_b)| {
                row_a
                    .iter_mut()
                    .zip(row_b.iter())
                    .for_each(|(a, b)| *a += *b)
            });
    }
}

///Matrix product. The implementation garuantees matrix compatibility at compile-time. If it compiles, it'll succeed.
///
/// # Commutativity
///
///Since matrix product is not commutative, the order matters.
/// ```
///# use matrix::Matrix;
/// let left = Matrix::from([[1, 2], [6, 7]]);
/// let right = Matrix::from([[3, 4], [5, 8]]);
/// assert_ne!(right.clone()*left.clone(), left*right)
/// ```
///
/// If the matrixes aren't square that also means you need to pay attention to their dimensions.
/// If the result of the product is of dimension *n\*p*, then the *left* matrix needs to be of dimension *n\*q* and the *right* one *q\*p*.
///
/// ```compile_fail
///# use matrix::Matrix;
/// let left = Matrix::from([[1, 2], [3, 4], [5, 6]]);
/// let right = Matrix::from([[9, 8, 7], [6, 5, 4]]);
/// right * left //<- this fails because right is of dimension 2*3 and left 3*2
/// ```
///
/// This however will compile.
/// ```
///# use matrix::Matrix;
/// let left = Matrix::from([[1, 2], [3, 4], [5, 6]]);
/// let right = Matrix::from([[9, 8, 7], [6, 5, 4]]);
/// assert_eq!(left*right, Matrix::from([[21, 18, 15], [51, 44, 37], [81, 70, 59]]));
///```
impl<C, const ROWS: usize, const COLS: usize, const Q: usize> Mul<Matrix<C, Q, COLS>>
    for Matrix<C, ROWS, Q>
where
    //clone may be more appropriate here, is there a way to require neither Copy nor Clone?
    C: Add + Mul<C, Output = C> + Sum + Clone,
{
    type Output = Matrix<C, ROWS, COLS>;
    fn mul(self, other: Matrix<C, Q, COLS>) -> Self::Output {
        let mut m: [[MaybeUninit<C>; COLS]; ROWS] = unsafe { MaybeUninit::uninit().assume_init() };
        for row in 0..ROWS {
            for col in 0..COLS {
                m[row][col].write(
                    self.data[row]
                        .iter()
                        .enumerate()
                        .map(|(j, a)| a.clone() * other.data[j][col].clone())
                        .sum(),
                );
            }
        }
        Matrix {
            //transmute_copy because transmute doesn't work for const generics yet
            data: unsafe { mem::transmute_copy::<_, [[C; COLS]; ROWS]>(&m) },
        }
    }
}

///Some functions for Matrix that have coefficients to have nil and neutral product values.
///This gives access to nil matrixes as well as the identity matrix
impl<C, const SIZE: usize> Matrix<C, SIZE, SIZE>
where
    C: One + Zero + Copy,
{
    ///Returns the identity matrix of size `SIZE`.
    /// Note that this often requires type annotation.
    ///
    /// # Example
    ///
    /// ```
    ///# use matrix::Matrix;
    /// let identity: Matrix<u8, 2, 2> = Matrix::identity();
    /// assert_eq!(identity, [[1, 0], [0, 1]].into());
    /// ```
    pub fn identity() -> Self {
        let mut m = Self::nil();
        for i in 0..SIZE {
            m.data[i][i] = C::one()
        }
        m
    }

    ///Returns the nil matrix of size `SIZE`.
    /// Note that this often requires type annotation.
    ///
    /// # Example
    ///
    /// ```
    ///# use matrix::Matrix;
    /// let identity: Matrix<u8, 2, 2> = Matrix::nil();
    /// assert_eq!(identity, [[0, 0], [0, 0]].into());
    /// ```
    pub fn nil() -> Self {
        [[C::zero(); SIZE]; SIZE].into()
    }
}

///Matrix internal manipulation operations
impl<'a, C: 'a, const SIZE: usize> Matrix<C, SIZE, SIZE>
where
    C: Clone + MulAssign<&'a C> + AddAssign<&'a C>,
{
    ///Permutation operation between `source` and `target` rows
    ///
    ///This can be understood as swapping the `source` row with the `target` one
    pub fn permute(&mut self, source: usize, target: usize) -> Result<(), Error> {
        if (target >= SIZE) | (source >= SIZE) {
            return Err(Error::OutOfBounds);
        } else {
            self.data.as_mut_slice().swap(source, target);
        }
        Ok(())
    }

    ///Dilation operation on row `row`.
    ///
    ///Line dilation is to multiply all coefficients of a row by a factor
    pub fn dilate(&mut self, row: usize, factor: &'a C) -> Result<(), Error> {
        match self.data.get_mut(row) {
            None => Err(Error::OutOfBounds),
            Some(line) => Ok(line.iter_mut().for_each(|c| *c *= factor)),
        }
    }
}

//Separated from the previous impl because of the need of HRTB
impl<C, const SIZE: usize> Matrix<C, SIZE, SIZE>
where
    for<'a> C: Clone + MulAssign<&'a C> + AddAssign<&'a C>,
{
    ///Transvection operation on row `source` with row `other` and `factor`
    ///
    ///Line transvection is to add a row to a source row for each coefficient
    pub fn transvect(&mut self, source: usize, other: usize) -> Result<(), Error> {
        if (other >= SIZE) | (source >= SIZE) {
            return Err(Error::OutOfBounds);
        } else if other == source {
            return Err(Error::WrongOperation);
        } else {
            let slices = self.data.as_mut_slice().split_at_mut(source.max(other));
            let (begin, end) = if source > other {
                (&mut slices.1[0], &mut slices.0[other])
            } else {
                (&mut slices.0[source], &mut slices.1[0])
            };
            begin
                .iter_mut()
                .enumerate()
                .for_each(|(i, c)| *c += &end[i]);
        }

        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid row: out of bounds")]
    OutOfBounds,
    #[error("there is an operation better suited for this")]
    WrongOperation,
}
