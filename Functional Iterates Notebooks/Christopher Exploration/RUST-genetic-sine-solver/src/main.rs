use core::f64;
use std::fmt;
use cgrustplot::plots::func_plot::function_plot;  // A custom plotting library, for plotting the output
use rand::distributions::{Uniform, Distribution};  // For random mutations
use rayon::prelude::*;  // For parallelization

// // Mathematical Helpers
const PI: f64 = std::f64::consts::PI;
const RANGE: (f64, f64) = (-2. * PI, 2. * PI); // Range over which to evaluate the function

fn factorial(num: usize) -> usize {
    let mut o = 1;
    for i in 1..=num { o *= i; }
    o
}

fn rand_vec(len: usize, range: (f64, f64)) -> Vec<f64> {
    if range.0 >= range.1 {
        (0..len).map(|_| range.0).collect()
    }
    else {

        let mut rng = rand::thread_rng();
        let dist = Uniform::new(range.0, range.1);

        (0..len)
            .map(|_| dist.sample(&mut rng))
            .collect()
    }
}

// Generate a function from a fourier series
fn fourier(coefs: &Vec<f64>) -> impl Fn(f64) -> f64 {
    let c = coefs.clone();
    
    move |x: f64| c
        .iter()
        .enumerate()
        .map(|(i, c)|
            c * (i as f64 * x).sin()
        ).sum::<f64>()
}

// Generate a function from a fourier series
fn cosine_fourier(coefs: &Vec<f64>) -> impl Fn(f64) -> f64 {
    let c = coefs.clone();
    
    move |x: f64| c
        .iter()
        .enumerate()
        .map(|(i, c)|
            c * (i as f64 * x).cos()
        ).sum::<f64>()
}

// Generate a function from a taylor series
fn taylor(coefs: &Vec<f64>) -> impl Fn(f64) -> f64 {
    let c = coefs.clone();

    move |x: f64| c
        .iter()
        .enumerate()
        .map(|(i, c)|
            c * x.powi(i as i32) / factorial(i) as f64
        ).sum::<f64>()
}

// Generate a function from a linear spline
fn linear(coefs: &Vec<f64>) -> impl Fn(f64) -> f64 {
    let c = coefs.clone();

    let n = c.len();
    let dx = (RANGE.1 - RANGE.0) / (n - 1) as f64;

    move |x: f64| {
        if x < RANGE.0 || x > RANGE.1 {
            0.
        } else {
            let left = ((x - RANGE.0) / dx).floor() as usize;
            let right = ((x - RANGE.0) / dx).ceil() as usize;

            if left == right {
                return c[left]
            }
            
            let weight = (x - (RANGE.0 + left as f64 * dx)) / dx;

            (1. - weight) * c[left] + weight * c[right]
        }
    }
}

// Use a reiman sum to integrate the loss of a function
fn loss<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(f: F, g: G) -> f64 {
    const RESOLUTION: f64 = 0.01;
    RESOLUTION * (((RANGE.0 / RESOLUTION) as i32)..(RANGE.1 / RESOLUTION) as i32)
    .map(|i| i as f64 * RESOLUTION)
    .map(|x| (g(x) - f(x)).powi(2))
    .sum::<f64>()
}

// Approximation definitions
#[derive(Debug, Clone)]
struct Approx {
    coefs: Vec<f64>,
    loss: Option<f64>,
    typ: char
}

impl fmt::Display for Approx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the coefficients to be able to be copied into wolfram
        let formatted_coefs_original: Vec<String> = self.coefs.iter().map(|i| format!("{:.16}, ", i)).collect();
        let mut formatted_coefs = vec![String::from("{")];
        formatted_coefs.extend(formatted_coefs_original);
        let fcl = formatted_coefs.len() - 1;
        formatted_coefs[fcl] = formatted_coefs[fcl][..(formatted_coefs[fcl].len()-2)].to_string();
        formatted_coefs.push(String::from("}"));
        let coefs_string = formatted_coefs.join("");
        write!(f, "\nBest Coefficients: \n{coefs_string}")
    }
}

impl Approx {
    fn new(coefs: Vec<f64>, typ: char) -> Self {
        Self {
            coefs,
            loss: None,
            typ,
        }
    }

    fn as_func(&self) -> Box<dyn Fn(f64) -> f64> {
        match self.typ {
            'f' => Box::new(fourier(&self.coefs)),
            't' => Box::new(taylor(&self.coefs)),
            'l' => Box::new(linear(&self.coefs)),
            'c' => Box::new(cosine_fourier(&self.coefs)),
            _ => Box::new(|x: f64| x),
        }
        
    }

    fn eval(&mut self, f: impl Fn(f64) -> f64) -> &Self {
        let func = self.as_func();

        self.loss = Some(loss(f, |x| func(func(x))));
        self

    }

    fn mutate(&self, temp: f64) -> Self {
        let adj: Vec<f64> = rand_vec(self.coefs.len(), (-temp, temp));
        let adj_coef: Vec<f64> = self.coefs
            .iter()
            .zip(
                adj.iter()
            )
            .map(|(a, b)| a + b)
            .collect();
        
        Self::new(adj_coef, self.typ.clone())
    }

    fn plot(&self) {
        // Use cgrustplot to plot the aprpoximation in the console
        let func = self.as_func();
        function_plot(func)
            .set_title(String::from("- Root Approximation - - -"))
            .set_domain(RANGE)
            .print();

        let func = self.as_func();
        let nested_func = move |x| func(func(x));
        function_plot(nested_func)
            .set_title(String::from("- Nested Root Approximation - - -"))
            .set_domain(RANGE)
            .print();
    }
}

fn eval_all<F: Fn(f64) -> f64 + Sized + Clone + std::marker::Sync>(v: &mut Vec<Approx>, f: &Box<F>) {
    v
    .par_iter_mut()
    .for_each(|ta|
        {ta.eval(f.clone());}
    );
}

fn sort_approx<F: Fn(f64) -> f64 + Sized + Clone + std::marker::Sync>(v: &mut Vec<Approx>, f: &Box<F>) {
    eval_all(v, f);

    v.sort_by(|a, b| a.loss.unwrap().partial_cmp(&b.loss.unwrap()).unwrap_or(std::cmp::Ordering::Equal))
}

fn random_approximations(num: usize, range: (f64, f64), coef_number: usize, typ: char) -> Vec<Approx> {
    (0..num).map(|_|
        Approx::new(rand_vec(coef_number, range), typ)
    )
    .collect()
}

/// Increases the number of fourier terms in a list of approximations
fn elongate(v: Vec<Approx>, n: usize, typ: char) -> Vec<Approx> {
    v
    .into_iter()
    .map(|ta| {
        let mut new_coefs = ta.coefs;
        new_coefs.extend((0..n).map(|_| 0.));
        Approx::new(new_coefs, typ)
    }).collect()
}

fn avg_loss(v: &Vec<Approx>) -> f64 {
    v.iter().map(|ta| ta.loss.unwrap_or(0.)).sum::<f64>() / v.len() as f64
}

fn genetic_optimize<F>(f: &Box<F>, approxes: Vec<Approx>, gens: u32, max_num: usize, min_num: usize) -> Vec<Approx>
where 
    F: Fn(f64) -> f64 + Sized + Clone + std::marker::Sync
{

    let mut v = approxes;

    // Zero mutation, for faster convergence of taylor series
    v.push(Approx::new(v[0].coefs.iter().map(|_| 0.).collect::<Vec<f64>>(), v[0].typ));

    let mut prev_loss = f64::INFINITY;
    let mut min_loss = f64::INFINITY;
    let mut t = 1.;

    for i in 0..gens {
        let debug_string = format!("\rGen {:03}/{gens} | temp: {t:.6} | Min Loss {min_loss:.4} | Improvement: {}    ", i + 1, prev_loss + min_loss);
        let debug_string2 = &debug_string[..(if debug_string.len() > 80 {80} else {debug_string.len()})];
        print!("{debug_string2}");
        
        // Generate mutations
        while v.len() < max_num {
            let extras: Vec<Approx> = v.par_iter().map(|ta| ta.mutate(t)).collect();
            v.extend(extras);
        }

        // Sort by loss
        sort_approx(&mut v, f);

        // Select the best
        v = v[..min_num].to_vec();

        prev_loss = min_loss;
        min_loss = v[0].loss.unwrap_or(0.);

        // Adjust temperature to allow for fast descent (0.1% improvement per step) while not staying at 0. 
        if min_loss == prev_loss {
            t *= 0.5;
        } else if min_loss / prev_loss > 0.99 {
            t *= 2.;
        }
        t *= 1.05;
    }

    print!("\n");

    v
}

// // EXAMPLES:
#[allow(dead_code)]
fn taylor_example<F>(f: &Box<F>) -> Approx
where 
    F: Fn(f64) -> f64 + Sized + Clone + std::marker::Sync
{
    let approx_type = 't';

    // Original random input
    let mut out: Vec<Approx> = random_approximations(2048, (-0.1, 0.1), 8, approx_type);
    
    // Optimize with different numbers of surviving approximations
    out = genetic_optimize(f, out, 500, 2048, 256);
    out = genetic_optimize(f, out, 2000, 256, 64);
    out = elongate(out, 8, approx_type);
    out = genetic_optimize(f, out, 500, 64, 16);
    out = genetic_optimize(f, out, 2000, 32, 8);
    
    // Find best approximation, output it
    sort_approx(&mut out, f);
    let best = (&out[0]).clone();
    println!("Best Coefficients: {best}");
    best.plot();
    best
}

#[allow(dead_code)]
fn fourier_example<F>(f: &Box<F>) -> Approx
where 
    F: Fn(f64) -> f64 + Sized + Clone + std::marker::Sync
{
    let approx_type = 'f';

    // Original random input
    let mut out: Vec<Approx> = random_approximations(256, (-10., 10.), 16, approx_type);
    
    // Optimize with different numbers of surviving approximations
    out = genetic_optimize(f, out, 500, 256, 64);
    out = elongate(out, 16, approx_type);
    out = genetic_optimize(f, out, 500, 256, 64);
    out = genetic_optimize(f, out, 500, 64, 16);
    
    // Find best approximation, output it
    sort_approx(&mut out, f);
    let best = (&out[0]).clone();
    println!("Best Coefficients: {best}");
    best.plot();
    best
}

#[allow(dead_code)]
fn linear_example<F>(f: &Box<F>) -> Approx
where 
    F: Fn(f64) -> f64 + Sized + Clone + std::marker::Sync
{
    let approx_type = 'l';

    // Original random input
    let mut out: Vec<Approx> = random_approximations(2048, RANGE, 256, approx_type);
    
    // Optimize with different numbers of surviving approximations
    out = genetic_optimize(f, out, 500, 2048, 256);
    out = genetic_optimize(f, out, 2000, 256, 16);
    out = genetic_optimize(f, out, 3000, 64, 8);
    
    // Find best approximation, output it
    sort_approx(&mut out, f);
    let best = (&out[0]).clone();
    println!("Best Coefficients: {best}");
    best.plot();
    best
}

fn main1() {
    let f = |x: f64| x.sin();

    fourier_example(&Box::new(f));
}

fn main() {
    let f = &Box::new(|x: f64| x.sin());

    let approx_type = 'f';

    // Original random input
    let mut out: Vec<Approx> = random_approximations(2048, (-10., 10.), 16, approx_type);
    
    // Optimize with different numbers of surviving approximations
    out = genetic_optimize(f, out, 2_00, 2048, 256);
    out = elongate(out, 16, approx_type);
    out = genetic_optimize(f, out, 5_00, 256, 64);
    out = elongate(out, 32, approx_type);
    out = genetic_optimize(f, out, 10_00, 256, 32);
    out = elongate(out, 64, approx_type);
    out = genetic_optimize(f, out, 1_00, 256, 16);
    
    // Find best approximation, output it
    sort_approx(&mut out, f);
    let best = (&out[0]).clone();
    println!("Best Coefficients: {best}");
    best.plot();
}
