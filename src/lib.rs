#![feature(maybe_uninit_extra)]
#![feature(array_methods)]
use num::traits::{One, Zero};
use std::convert::From;
use std::iter::Sum;
use std::mem::{self, MaybeUninit};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::slice::{Iter, IterMut};
use thiserror::Error;

#[derive(Debug, PartialEq, Eq)]
///C is the type of the coefficients of the Matrix
///ROWS is the number of lines & COLS the number of columns
pub struct Matrix<C, const ROWS: usize, const COLS: usize> {
    data: [[C; COLS]; ROWS],
}

impl<C, const ROWS: usize, const COLS: usize> Matrix<C, ROWS, COLS> {
    ///Returns an iterator of all lines of the matrix.
    pub fn get_lines(&self) -> Iter<[C; COLS]> {
        self.data.iter()
    }

    pub fn get_mut_lines(&mut self) -> IterMut<[C; COLS]> {
        self.data.iter_mut()
    }

    ///Returns the n-1 line. If `index` is out of range then None.
    pub fn get_line(&self, index: usize) -> Option<&[C; COLS]> {
        self.data.get(index)
    }

    pub fn get_mut_line(&mut self, index: usize) -> Option<&mut [C; COLS]> {
        self.data.get_mut(index)
    }

    ///Access a coefficient. None if the indexes are greated than the size
    pub fn get(&self, row: usize, col: usize) -> Option<&C> {
        match self.data.get(row) {
            None => None,
            Some(line) => line.get(col),
        }
    }

    ///Mutable access a coefficient. None if the indexes are greated than the size
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

///Multiplication by a coefficient. Can never fail, works for all matrices.
impl<'a, C: 'a, const ROWS: usize, const COLS: usize> MulAssign<C> for Matrix<C, ROWS, COLS>
where
    C: MulAssign + Copy,
{
    fn mul_assign(&mut self, coef: C) {
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

///Matrix product. The implementation garuantees success by checking compatibility with ROWS and COLS bounds
///Creates a new Martix
impl<C, const ROWS: usize, const COLS: usize, const Q: usize> Mul<Matrix<C, Q, COLS>>
    for Matrix<C, ROWS, Q>
where
    //clone may be more appropriate here, is there a way not to require neither Copy nor Clone?
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
    ///Returns the identity matrix of size `SIZE`
    pub fn identity() -> Self {
        let mut m = Self::nil();
        for i in 0..SIZE {
            m.data[i][i] = C::one()
        }
        m
    }

    ///Returns the nil matrix of size `SIZE`
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
