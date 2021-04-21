#lang racket

; ML with neural networks is a lot of matrix multiplication, because
; transformation on the network layers can be seen as matrixes and each
; individual layer as a collection of vectors (i.e.: from previous layers,
; weights, biases etc)
(require math/matrix)

;; Common functions used for activation

; One of those is chosen to be applied to the output of each neuron in each
; layer (i.e.: map each element in the vector)

; We use the notation function~ to denote the derivative of function

(define (sigmoid x) (/ (exp x) (+ (exp x) 1)))
(define (sigmoid~ x) (* (sigmoid x) (- 1 (sigmoid x))))
(define (tanh~ x) (/ 1 (expt (cosh x) 2)))
(define (relu x) (max x 0))
(define (relu~ x) (if (> x 0) 1 0))

(define activation sigmoid)
(define activation~ sigmoid~)

; Calculates output and intermediate output for each layer, feeds into the next
; layer etc
(define (feedforward input weights biases)
  (if (empty? weights)
      (list)

      ; Compute the intermediate output (z) and the final (out)
      ;
      ; Here we multiply a weights vector with the input (or output from
      ; previous later) and sum that with the biases. We call this z. Then we
      ; run the activation function for each one of those values and call the
      ; new vector out.
      (let* ([z (matrix+ (matrix* (first weights) input) (first biases))]
             [out (matrix-map activation z)])
        ; Return z and out + the result from the next layer. If the next is the last it
        ; will be the final output, otherwise the intermediate output (z) for
        ; that layer.
        ;
        ; This makes the final return a list in the form:
        ;
        ; (z1 o1 z2 o2 ... zn on)
        (cons
         ; 1. The values we just calculated
         (list z out)

         ; 2. Compute the next layer (using the output from this one)
         (feedforward out (rest weights) (rest biases))))))

(define (output-error z actual expected)
  (matrix-map * (matrix- actual expected) (matrix-map activation~ z)))

(define (propagate-error outputs weights next-error)
  (if (empty? outputs)
      (list next-error)
      (cons
       next-error

       (propagate-error
        (rest outputs)
        (rest weights)
        (matrix-map *
                    (matrix* (first weights) next-error)
                    (matrix-map activation~ (first (first outputs))))))))

; NOTE: outputs, weights need to be (reverse)d
(define (backprop weights outputs expected-output)
  (propagate-error (rest outputs)
                   weights
                   (let ([out (first outputs)])
                     (output-error (first out) (second out) expected-output))))

(define (adjust-weights weight out err epsilon)
  (let-values ([(w h) (matrix-shape weight)])
    (build-matrix w h
                  (lambda (j k) (-
                                 (matrix-ref weight j k)
                                 (* (matrix-ref out k 0) (matrix-ref err j 0) epsilon))))))

(define (adjust-parameters weights biases outputs errors)
  (let ([epsilon 1])
    (if (empty? outputs)
        (list)
        (cons
         (list
          (adjust-weights (first weights) (second (first outputs)) (first errors) epsilon)
          (matrix- (first biases) (matrix-scale (first errors) epsilon)))

         (adjust-parameters
          (rest weights)
          (rest biases)
          (rest outputs)
          (rest errors))))))

(define (gradient-descent weights biases inputs expected-output (n 1))
  (let* ([outputs (feedforward inputs weights biases)]
         [errors (reverse (backprop (reverse weights) (reverse outputs) expected-output))]
         [parameters (reverse (adjust-parameters weights biases outputs errors))]
         [weights (map (lambda (x) (first x)) parameters)]
         [biases (map (lambda (x) (second x)) parameters)])
    (if (<= n 1)
        (values weights biases)
        (gradient-descent weights biases inputs expected-output (sub1 n)))))

(define (matrix-size m)
  (call-with-values (lambda () (matrix-shape m)) *))

(define (parameters-in-net weights biases)
  (+ (* (length weights) (matrix-size (first weights)))
     (* (length biases) (matrix-size (first biases)))))

(let* ([weights (list (matrix (((random) (random))
                               ((random) (random))))
                      (matrix (((random) (random))
                               ((random) (random))))
                      (matrix (((random) (random))
                               ((random) (random))))
                      (matrix (((random) (random))
                               ((random) (random))))
                      (matrix (((random) (random))
                               ((random) (random))))
                      )]
       [biases (list (col-matrix ((random) (random)))
                     (col-matrix ((random) (random)))
                     (col-matrix ((random) (random)))
                     (col-matrix ((random) (random)))
                     (col-matrix ((random) (random)))
                     )]
       [inputs (col-matrix (1 1))]
       [expected-output (col-matrix (0.5 0.25))])
  (begin
    (display (format "Training model with ~a parameters" (parameters-in-net weights biases)))
    (newline)

    (let-values ([(weights biases) (gradient-descent weights biases inputs expected-output 256)])
      (begin
        (display (second (last (feedforward inputs weights biases)))) (newline)))))
